"""LLM interface: fact extraction and text embedding via DeepSeek.

Uses the OpenAI-compatible API exposed by DeepSeek for both chat completion
(extract_facts) and embeddings (embed / embed_batch).
"""

import logging

import json

logger = logging.getLogger(__name__)

import openai

from src import config


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_MAX_TOPICS = str(config.MAX_TOPICS_PER_FACT)

_EXTRACTION_PROMPT = (
    "You are a fact extractor for a long-term memory system. You are given a\n"
    "transcript of a conversation between a user and an assistant. Extract durable,\n"
    "atomic facts worth remembering, and assign each fact to the subject(s) it is\n"
    "about.\n"
    "\n"
    'Output ONLY a JSON array of objects, each {"fact": "...", "topics": ["..."]}.\n'
    "No prose, no explanation, no markdown code fences. If there are no facts worth\n"
    "storing, output [].\n"
    "\n"
    "WHAT TO EXTRACT (unchanged rules):\n"
    "- Durable facts: preferences, background, ongoing projects, relationships, goals,\n"
    "  constraints, decisions, stable properties of things in the user's world.\n"
    "- NOT transient/conversational content: greetings, the assistant's suggestions,\n"
    "  questions, one-off task chatter.\n"
    "- Facts about the user and their world, not about the assistant.\n"
    "\n"
    "HOW TO WRITE EACH FACT (unchanged rules):\n"
    "- ATOMIC: exactly one claim. Never join claims with \"and\".\n"
    "- SELF-CONTAINED: makes sense months later with no access to this conversation;\n"
    "  resolve pronouns and vague references to concrete nouns.\n"
    "- Present-tense, complete statement.\n"
    '- Refer to the user consistently (e.g. "The user ..."). Do not use their name\n'
    "  unless the name itself is the fact.\n"
    "- State only what the transcript supports. Do not infer or speculate.\n"
    "\n"
    "HOW TO ASSIGN THE TOPIC(S) - the subject the fact is really about:\n"
    "- The topic is the PRIMARY ENTITY the fact is genuinely about: a project, a\n"
    "  person, a physical object, a place, an organisation, an activity - whatever it\n"
    "  actually concerns.\n"
    '- IGNORE the grammatical subject. Almost every fact is phrased "The user ...", so\n'
    '  "starts with the user" does NOT mean the fact is about the user. Look past the\n'
    "  phrasing to the real subject.\n"
    "- TEST: the real subject is the thing the fact stays about if everything else\n"
    "  changed.\n"
    '    * "The user uses systemd-run to run the collector"  -> the PROJECT the\n'
    "      collector belongs to (kill the project and the fact is meaningless). NOT\n"
    "      the user.\n"
    '    * "The cypress is on the south-facing balcony"  -> the CYPRESS (move\n'
    "      apartments, still a fact about the tree). NOT the user, NOT a project.\n"
    '    * "The user has Multiple Sclerosis"  -> the USER (true about the person\n'
    "      regardless of any project).\n"
    "- PERSON BUCKET: facts that are durably true about the PERSON across all their\n"
    "  work - who they are, where they're based, standing cross-project preferences -\n"
    '  all share ONE topic: "user profile". Do NOT split these into "user\'s health",\n'
    '  "user\'s location", etc.\n'
    '    * Joins "user profile" ONLY if true about the person independent of any\n'
    "      project or object.\n"
    '    * "Prefers minimal, inspectable implementations" -> user profile.\n'
    '    * "Uses systemd-run" -> the project, NOT user profile.\n'
    "      The person bucket is not a catch-all for anything phrased \"The user ...\".\n"
    "\n"
    "MULTIPLE TOPICS - the exception, not the default:\n"
    "- Give exactly ONE topic unless the fact is substantively about a SECOND subject\n"
    "  such that leaving it out would lose real information.\n"
    '- "Relatable to" is not enough - it must genuinely BELONG to both.\n'
    f"- At most {_MAX_TOPICS} topics. Prefer one. The first topic is the primary subject.\n"
    "- Only co-equal (peer) subjects. Do not add a topic to express a relationship\n"
    "  between subjects.\n"
    "\n"
    "CONSISTENCY: use the SAME topic phrase for the same subject across all facts in\n"
    "this conversation, so the same project/person/object is not named two different\n"
    "ways.\n"
    "\n"
    "OWNERSHIP AND RELATIONSHIP FACTS - file under the THING, emit a relation:\n"
    "- A fact asserting the user has/owns/created/built/works-on a named thing is\n"
    "  primarily a fact ABOUT THAT THING - that it exists and whose it is - NOT a\n"
    "  fact about the person.\n"
    '    * "The user has a project called Gemory" -> topic: "Gemory" (NOT user profile).\n'
    '    * "The user created MS Navigator" -> topic: "MS Navigator" (NOT user profile).\n'
    '    * "The user owns a bald cypress" -> topic: "bald cypress tree".\n'
    "- When a fact asserts a link between two subjects, add a \"relates\" field\n"
    "  with the endpoints. Both endpoints must be the SAME topic phrases used\n"
    "  elsewhere in this conversation.\n"
    '    {"from": "user profile", "to": "Gemory"}\n'
    '- "relates" is OPTIONAL and usually absent. Only include it when the fact\n'
    "  genuinely ASSERTS A LINK between two subjects - not merely mentions\n"
    "  another subject in passing.\n"
    "- RELATION vs MULTI-TOPIC (critical - pick ONE):\n"
    "  * Multi-topic (topics: [A, B]) = the fact is ABOUT both subjects.\n"
    "  * Relation (relates: [{from: A, to: B}]) = the fact asserts a LINK between them.\n"
    "  * For ownership/existence facts, use ONE topic (the thing) + ONE relation -\n"
    '    NEVER two topics. "The user has a project called Gemory" -> one topic\n'
    "    (Gemory) + one relation (user profile -> Gemory). Do NOT file under user\n"
    "    profile too.\n"
    "\n"
    'Output format:\n'
    '[{"fact": "...", "topics": ["primary subject"]},\n'
    ' {"fact": "...", "topics": ["primary subject"],\n'
    '  "relates": [{"from": "user profile", "to": "Gemory"}]}]'
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_extraction_response(content: str) -> list[dict]:
    """Parse a JSON array of ``{fact, topics}`` objects from the LLM response.

    Backward-compatible:
    - Old ``{"fact": "...", "topic": "..."}`` -> ``{"fact": "...", "topics": [...]}``.
    - Plain strings -> ``{"fact": s, "topics": []}``.

    Strategy
    --------
    1. Try :func:`json.loads` on the raw content.
    2. Strip markdown code fences and find the outermost ``[…]`` bracket
       pair, then try ``json.loads`` on that substring.
    3. If all attempts fail, raise :class:`ValueError` with the raw content.
    """
    max_topics = config.MAX_TOPICS_PER_FACT

    def _normalize(item):
        """Convert a single parsed item to {fact, topics} format."""
        if isinstance(item, str):
            return {"fact": item, "topics": []}
        if isinstance(item, dict):
            fact = str(item.get("fact", ""))
            topics_raw = item.get("topics", [])
            # Backward compat: old single-topic format.
            if not topics_raw and "topic" in item:
                t = item["topic"]
                topics_raw = [t] if t else []
            # Ensure it is a list.
            if not isinstance(topics_raw, list):
                topics_raw = [str(topics_raw)]
            # Strip empties and cap.
            topics = [str(t).strip() for t in topics_raw if t and str(t).strip()]
            topics = topics[:max_topics]
            relates_raw = item.get("relates", [])
            if not isinstance(relates_raw, list):
                relates_raw = []
            relates = []
            for rel in relates_raw:
                if isinstance(rel, dict) and "from" in rel and "to" in rel:
                    relates.append({
                        "from": str(rel.get("from", "")),
                        "to": str(rel.get("to", "")),
                    })
            return {"fact": fact, "topics": topics, "relates": relates}
        return {"fact": str(item), "topics": []}

    # ── Attempt 1: direct JSON parse ──────────────────────────────────
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [_normalize(item) for item in data]
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: strip fences, locate outermost [ … ] ───────────────
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        if candidate != content:  # avoid repeating the direct parse
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return [_normalize(item) for item in data]
            except json.JSONDecodeError:
                pass

    # ── All attempts exhausted ────────────────────────────────────────
    logger.error(
        "Could not parse LLM response as JSON array. Raw response: %s",
        content[:500],
    )
    raise ValueError(
        "Could not parse LLM response as a JSON array of {fact, topics} objects.\n"
        f"Raw response:\n{content}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(transcript: str) -> list[dict]:
    """Extract atomic, durable facts from a conversation *transcript*.

    Returns a list of dicts with keys ``"fact"`` and ``"topics"`` (a list).
    Raises :class:`ValueError` if the LLM response cannot be parsed.

    Raises
    ------
    openai.AuthenticationError
        Invalid or missing API key.
    openai.RateLimitError
        API rate limit exceeded.
    ValueError
        Response could not be parsed as a JSON array of {fact, topic} objects.
    """
    logger.info("Extracting facts from transcript (%d chars)", len(transcript))
    logger.info(
        "LLM request: model=%s base_url=%s system_prompt(%d_chars) transcript=%r",
        config.DEEPSEEK_CHAT_MODEL,
        config.DEEPSEEK_BASE_URL,
        len(_EXTRACTION_PROMPT),
        transcript,
    )

    client = openai.OpenAI(
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
    except Exception:
        logger.exception("Fact extraction failed")
        raise

    raw = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    usage = response.usage
    if usage:
        logger.info(
            "LLM response: tokens(prompt=%d, completion=%d, total=%d) "
            "finish=%s raw=%r",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            finish_reason,
            raw if raw else None,
        )
    else:
        logger.info(
            "LLM response: finish=%s raw=%r",
            finish_reason,
            raw if raw else None,
        )

    facts = _parse_extraction_response(raw)
    topics_count = sum(1 for f in facts if f.get("topics"))
    logger.info("Extracted %d facts (%d with topics)", len(facts), topics_count)
    return facts


def embed(text: str) -> list[float]:
    """Embed a single *text* string and return its vector."""
    logger.info("Embedding text (%d chars)", len(text))
    client = openai.OpenAI(
        base_url=config.EMBEDDING_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
    )

    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[text],
    )

    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple *texts* and return a list of vectors.

    The output order matches the input order.
    """
    if not texts:
        logger.info("embed_batch called with empty input, returning []")
        return []

    logger.info("Embedding %d texts", len(texts))
    client = openai.OpenAI(
        base_url=config.EMBEDDING_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
    )

    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )

    # Sort by index to ensure input-order fidelity.
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [d.embedding for d in sorted_data]


# ---------------------------------------------------------------------------
# Cluster summarization
# ---------------------------------------------------------------------------

def summarize_cluster(member_facts: list[str]) -> dict[str, str]:
    """Ask the LLM to produce a label and summary for a cluster of facts.

    Returns a dict with keys ``"label"`` and ``"summary"``.

    Raises :class:`ValueError` if the LLM response cannot be parsed.
    """
    logger.info("Summarizing cluster of %d items", len(member_facts))

    facts_text = "\n".join(f"- {f}" for f in member_facts)

    prompt = (
        "You are a knowledge consolidator for a long-term memory system. "
        "You are given a set of related facts about a user. "
        "Your job is to identify the common theme and write a concise summary.\n"
        "\n"
        "Output ONLY a JSON object with exactly two keys: \"label\" and \"summary\". "
        "No prose, no explanation, no markdown code fences.\n"
        "\n"
        "Rules:\n"
        "- \"label\": a SHORT theme label (3-6 words), something you would scan in a list.\n"
        "- \"summary\": a 1-2 sentence description of what these facts have in common, "
        "written to be read months later with no other context. Self-contained, no "
        "dangling pronouns.\n"
        "- Do NOT invent facts not supported by the member facts. The abstraction "
        "describes the cluster; it does not add new claims.\n"
        "- If the facts do not share a clear common theme, use a label like "
        "\"Miscellaneous facts\" and state honestly in the summary that no strong "
        "theme emerged.\n"
        "\n"
        "Format: {\"label\": \"theme label\", \"summary\": \"1-2 sentence summary\"}"
    )

    client = openai.OpenAI(
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": facts_text},
            ],
        )
    except Exception:
        logger.exception("Cluster summarization failed")
        raise

    raw = response.choices[0].message.content
    usage = response.usage
    finish_reason = response.choices[0].finish_reason
    if usage:
        logger.info(
            "Summarization response: tokens(prompt=%d, completion=%d, total=%d) "
            "finish=%s raw=%r",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            finish_reason,
            raw if raw else None,
        )
    else:
        logger.info(
            "Summarization response: finish=%s raw=%r",
            finish_reason,
            raw if raw else None,
        )

    parsed = _parse_summarize_response(raw)
    logger.info(
        "Summarized cluster: label=%r, summary=%r",
        parsed.get("label"),
        parsed.get("summary", "")[:80],
    )
    return parsed


def _parse_summarize_response(content: str) -> dict[str, str]:
    """Parse the LLM response into a ``{label, summary}`` dict."""
    import json

    # Attempt 1: direct JSON parse
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "label" in data and "summary" in data:
            return {"label": str(data["label"]), "summary": str(data["summary"])}
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip fences, locate outermost { ... }
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        if candidate != content:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "label" in data and "summary" in data:
                    return {"label": str(data["label"]), "summary": str(data["summary"])}
            except json.JSONDecodeError:
                pass

    logger.error(
        "Could not parse summarization response. Raw: %s", content[:500],
    )
    raise ValueError(
        f"Could not parse LLM response as a JSON object with label+summary.\n"
        f"Raw response:\n{content}"
    )


# ---------------------------------------------------------------------------
# LLM clustering
# ---------------------------------------------------------------------------

def cluster_by_llm(node_summaries: list[dict[str, str]]) -> list[set[int]]:
    """Ask the LLM to group nodes into thematic clusters.

    *node_summaries* is a list of dicts with keys:
      - ``"index"``: integer position (0-based, for output mapping)
      - ``"label"``: short label of the node
      - ``"summary"``: summary/content of the node

    Returns a list of clusters, each a set of indices into *node_summaries*.
    Nodes not grouped are not included in any cluster.

    Raises :class:`ValueError` if the LLM response cannot be parsed.
    """
    # Build the input string.
    lines: list[str] = []
    for item in node_summaries:
        lines.append(f"{item['index']}: [{item['label']}] {item['summary']}")
    input_text = "\n".join(lines)

    prompt = (
        "You are a knowledge organizer. You are given a list of items, each with "
        "a label and a short summary. Your job is to group items that belong "
        "together into thematic clusters.\n"
        "\n"
        "Output ONLY a JSON array of arrays of integers. Each inner array "
        "contains the indices of items that form one cluster. Items that do not "
        "belong to any cluster should not appear in any group.\n"
        "\n"
        "Rules:\n"
        "- Group items that share a common theme, subject, or category -- they\n"
        "  must be the SAME KIND OF THING.\n"
        "- Do NOT force items together if they are genuinely unrelated.\n"
        "- Most sets of items do NOT share a theme. If the items are genuinely\n"
        "  unrelated, return [] (empty array). This is the CORRECT answer for\n"
        "  unrelated items. Do NOT force groupings.\n"
        "- Only group items when a genuine common theme exists and is clearly\n"
        "  supported by the items.\n"
        "- A cluster should have at least 2 items.\n"
        "- Do not invent themes not supported by the items.\n"
        "\n"
        "SAME KIND OF THING -- a legitimate grouping criterion:\n"
        "- Items can share a theme because they are the SAME KIND OF THING,\n"
        "  even when their purpose, technology, and domain differ entirely.\n"
        '  * Four items described as "a zero-data static MS information site",\n'
        '    "an agentic memory system using a graph of topics", "a read-only\n'
        '    TUI for exploring a local Honcho install", and "a command-line tool\n'
        '    for managing Honcho" -> theme: "software projects the user builds".\n'
        '  * "a bald cypress bonsai", "a ficus bonsai", "a juniper bonsai" ->\n'
        '    theme: "bonsai trees the user tends".\n'
        "- The shared KIND -- what the items ARE -- is what makes the theme real,\n"
        "  not whether they serve the same purpose or use the same technology.\n"
        "- A shared relationship to the user (all things the user owns/made/has)\n"
        "  is NOT a kind -- it merely restates that the items exist in one\n"
        "  person's memory. \"All the user's projects\" says nothing about what\n"
        "  any of them are. \"Software projects\" says something: they are all\n"
        "  software.\n"
        "\n"
        "CRITICAL RULES -- confabulation prevention:\n"
        '- "Things the user made/has/did" is NOT a theme -- it is a restatement\n'
        "  of the fact that these items exist in one person's memory, and applies\n"
        "  to literally everything in the graph. Never group on this basis.\n"
        "- An enumeration of projects the user created is NOT a theme -- it is a\n"
        "  list, not a category.\n"
        "- Most sets of items do NOT share a theme. If items are genuinely\n"
        "  unrelated to each other, return [] (empty array). This is the CORRECT\n"
        "  and EXPECTED answer. Do not force a grouping.\n"
        "- Only group items when a genuine common theme exists and is clearly\n"
        "  supported. A theme means the items are the SAME KIND OF THING (e.g.,\n"
        "  all are software projects, all are physical objects), not merely that\n"
        "  one person is associated with all of them.\n"
        "- If you are uncertain, return [] -- it is better to miss a real theme\n"
        "  than to invent a false one.\n"
        "\n"
        "Example: if items 0, 2, 5 are about coding tools and items 1, 4 are "
        "about photography, output: [[0, 2, 5], [1, 4]]\n"
        "\n"
        "Output format: [[0, 2, 5], [1, 4]]"
    )

    client = openai.OpenAI(
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
        )
    except Exception:
        logger.exception("LLM clustering failed")
        raise

    raw = response.choices[0].message.content
    usage = response.usage
    if usage:
        logger.info(
            "LLM clustering: tokens(prompt=%d, completion=%d, total=%d) raw=%r",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
            raw if raw else None,
        )
    else:
        logger.info("LLM clustering: raw=%r", raw if raw else None)

    clusters = _parse_cluster_response(raw)
    logger.info("LLM clustering produced %d groups", len(clusters))
    return clusters


def _parse_cluster_response(content: str) -> list[set[int]]:
    """Parse LLM cluster response into list of index sets."""
    import json

    # Attempt 1: direct JSON parse
    try:
        data = json.loads(content)
        return _validate_clusters(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: strip fences
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        data = json.loads(cleaned)
        return _validate_clusters(data)
    except (json.JSONDecodeError, ValueError):
        pass

    logger.error(
        "Could not parse LLM cluster response. Raw: %s", content[:500],
    )
    raise ValueError(
        "Could not parse LLM cluster response as JSON array of arrays.\n"
        f"Raw response:\n{content}"
    )


def _validate_clusters(data) -> list[set[int]]:
    """Validate and convert raw parsed data to list of sets of ints."""
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array")
    result: list[set[int]] = []
    for item in data:
        if not isinstance(item, list):
            raise ValueError("Expected array of arrays")
        result.append(set(int(i) for i in item))
    return result

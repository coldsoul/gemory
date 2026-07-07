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

_EXTRACTION_PROMPT = (
    "You are a fact extractor for a long-term memory system. You are given a\n"
    "transcript of a conversation between a user and an assistant. Your job is to\n"
    "extract durable, atomic facts worth remembering about the user and their world.\n"
    "\n"
    "Output ONLY a JSON array of objects. No prose, no explanation, no markdown code\n"
    "fences. If there are no facts worth storing, output [].\n"
    "\n"
    "Each object must have exactly two keys: \"fact\" and \"topic\".\n"
    "- \"fact\": the atomic fact statement (see rules below).\n"
    "- \"topic\": a short (2-5 word) noun phrase naming the SUBJECT the fact is\n"
    "  about -- the project, entity, or area -- not a restatement of the fact.\n"
    "  Good: fact \"The user implemented the testing harness for their memory system\"\n"
    "        -> topic \"Gemory memory system\".\n"
    "  Use the SAME topic phrase for all facts about the same subject within this\n"
    "  conversation (be consistent). Prefer a concrete project/entity name if one\n"
    "  is present (e.g. \"Gemory\", \"Sofia transit tracker\").\n"
    "  If a fact does not clearly belong to any subject, use an EMPTY topic (\"\").\n"
    "  Do not force one. The topic must be supported by the conversation; do NOT\n"
    "  invent a subject not present.\n"
    "\n"
    "Rules for what to extract:\n"
    "- Extract DURABLE facts: things likely to remain true and be useful in future\n"
    "  conversations (preferences, background, ongoing projects, relationships,\n"
    "  goals, constraints, decisions, stable opinions).\n"
    "- Do NOT extract transient or conversational content: greetings, the assistant's\n"
    "  suggestions, questions, one-off task details, or anything that only matters\n"
    "  within this single conversation.\n"
    "- Extract facts about the USER and their world, not about the assistant.\n"
    "\n"
    "Rules for HOW to write each fact (this is the most important part):\n"
    "- Each fact must be ATOMIC: exactly one claim. Never combine claims with \"and\".\n"
    '  Bad:  "The user uses uv and works on a VPS and likes bonsai"\n'
    '  Good: "The user uses uv"\n'
    '        "The user works on a VPS"\n'
    '        "The user likes bonsai"\n'
    "- Each fact must be SELF-CONTAINED: it must make sense on its own, months later,\n"
    "  with no access to this conversation. Resolve pronouns and vague references to\n"
    "  concrete nouns.\n"
    '  Bad:  "He wants to build it in Python"\n'
    '  Good: "The user wants to build the memory system in Python"\n'
    "- Write each fact as a complete, present-tense statement.\n"
    "- State only what the transcript supports. Do not infer, speculate, or embellish.\n"
    "\n"
    'Output format: [{"fact": "fact one", "topic": "topic name"}, ...]'
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_extraction_response(content: str) -> list[dict[str, str]]:
    """Parse a JSON array of ``{fact, topic}`` objects from the LLM response.

    Backward-compatible: if the model returns a plain array of strings,
    each string is wrapped into ``{"fact": s, "topic": ""}``.

    Strategy
    --------
    1. Try :func:`json.loads` on the raw content.
    2. Strip markdown code fences and find the outermost ``[…]`` bracket
       pair, then try ``json.loads`` on that substring.
    3. If all attempts fail, raise :class:`ValueError` with the raw content.
    """

    def _normalize(item):
        """Convert a single parsed item to {fact, topic} format."""
        if isinstance(item, str):
            return {"fact": item, "topic": ""}
        if isinstance(item, dict):
            return {
                "fact": str(item.get("fact", "")),
                "topic": str(item.get("topic", "")),
            }
        return {"fact": str(item), "topic": ""}

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
        "Could not parse LLM response as a JSON array of {fact, topic} objects.\n"
        f"Raw response:\n{content}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(transcript: str) -> list[dict[str, str]]:
    """Extract atomic, durable facts from a conversation *transcript*.

    Returns a list of dicts with keys ``"fact"`` and ``"topic"``.
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
    topics_count = sum(1 for f in facts if f.get("topic"))
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
    logger.info("Summarizing cluster of %d facts", len(member_facts))

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
        "- Group items that share a common theme, subject, or category.\n"
        "- Do NOT force items together if they are genuinely unrelated.\n"
        "- Most sets of items do NOT share a theme. If the items are genuinely\n"
        "  unrelated, return [] (empty array). This is the CORRECT answer for\n"
        "  unrelated items. Do NOT force groupings.\n"
        "- Only group items when a genuine common theme exists and is clearly\n"
        "  supported by the items.\n"
        "- A cluster should have at least 2 items.\n"
        "- Do not invent themes not supported by the items.\n"
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

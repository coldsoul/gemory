"""LLM interface: fact extraction and text embedding via DeepSeek.

Uses the OpenAI-compatible API exposed by DeepSeek for both chat completion
(extract_facts) and embeddings (embed / embed_batch).
"""

import logging

import json

logger = logging.getLogger(__name__)

import openai

from gemory import config


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = (
    "You are a fact extractor for a long-term memory system. You are given a\n"
    "transcript of a conversation between a user and an assistant. Your job is to\n"
    "extract durable, atomic facts worth remembering about the user and their world.\n"
    "\n"
    "Output ONLY a JSON array of strings. No prose, no explanation, no markdown code\n"
    "fences. If there are no facts worth storing, output [].\n"
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
    "- Use a consistent way of referring to the user across facts (e.g. always \"The\n"
    '  user ..."). Do not use their name unless it is itself the fact being stored.\n'
    "- State only what the transcript supports. Do not infer, speculate, or embellish.\n"
    "  If something is uncertain or hypothetical, either omit it or state the\n"
    "  uncertainty explicitly as part of the fact.\n"
    "\n"
    'Output format: ["fact one", "fact two", ...]'
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_facts_response(content: str) -> list[str]:
    """Parse a JSON array of strings from the LLM response *content*.

    Strategy
    --------
    1. Try :func:`json.loads` on the raw content.
    2. Strip markdown code fences (`` ```json `` / `` ``` ``) and find the
       outermost ``[…]`` bracket pair, then try ``json.loads`` on that
       substring.
    3. If all attempts fail, raise :class:`ValueError` with the raw content.
    """
    # ── Attempt 1: direct JSON parse ──────────────────────────────────
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(item) for item in data]
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
        elif cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        if candidate != content:  # avoid repeating the direct parse
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return [str(item) for item in data]
            except json.JSONDecodeError:
                pass

    # ── All attempts exhausted ────────────────────────────────────────
    logger.error(
        "Could not parse LLM response as JSON array. Raw response: %s",
        content[:500],
    )
    raise ValueError(
        f"Could not parse LLM response as a JSON array of strings.\n"
        f"Raw response:\n{content}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(transcript: str) -> list[str]:
    """Extract atomic, durable facts from a conversation *transcript*.

    Returns a list of fact strings.  Raises :class:`ValueError` if the
    LLM response cannot be parsed.

    Raises
    ------
    openai.AuthenticationError
        Invalid or missing API key.
    openai.RateLimitError
        API rate limit exceeded.
    ValueError
        Response could not be parsed as a JSON array of strings.
    """
    logger.info("Extracting facts from transcript (%d chars)", len(transcript))

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
    facts = _parse_facts_response(raw)
    logger.info("Extracted %d facts", len(facts))
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

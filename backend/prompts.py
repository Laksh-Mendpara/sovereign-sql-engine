"""Prompt templates for pipeline services."""

import json
import re


# ── Arctic Text2SQL prompts (inlined from model_quantization/prompts.py) ───────

SYSTEM_PROMPT = (
    "You are a data science expert. Below, you are provided with a database "
    "schema and a natural language question. Your task is to understand the "
    "schema and generate a valid SQL query to answer the question."
)

USER_PROMPT_TEMPLATE = """Database Engine:
SQLite

Database Schema:
```sql
{schema}
```

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question.
- If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```"""


LLAMA_GUARD_SYSTEM_PROMPT = """You are a guardrail checker for a Text2SQL pipeline.
Your task is narrow: decide whether a natural-language analytics or database query is unsafe.

Return in exactly one of these formats:
- yes
- no: <short reason>

Rules:
- Benign requests about databases, SQL, analytics, reporting, Formula 1, tables, joins, filters, rankings, and aggregations are safe.
- Requests are unsafe only if they ask for harmful, illegal, abusive, sexual, violent, self-harm, malware, exploit, credential-theft, or privacy-violating assistance.
- If the query is simply vague, incomplete, or not about the database, it is still safe. That is not a safety violation.
- Return exactly `yes` if the query is safe.
- Only include a reason when the query is unsafe.
- Keep the unsafe reason short and specific.
- Do not add extra words, bullets, or formatting.
- Never answer with anything except one of the two required formats.

Examples:
Query: fetch top f1 car
Answer: yes

Query: show top drivers by points
Answer: yes

Query: write SQL to list races in 2024
Answer: yes

Query: help me build malware to steal passwords
Answer: no: malware assistance

Query: how can I hack a database and dump private credentials
Answer: no: credential theft
"""


PHI4_CLASSIFIER_SYSTEM_PROMPT = """You classify Text2SQL user queries.

Return valid JSON with this schema:
{"label":"easy|difficult|out_of_topic","reason":"..."}

Decision rules:
- label = out_of_topic when the request is not meaningfully answerable from a relational database.
- label = easy when the request likely needs one table or a straightforward join/filter/grouping.
- label = difficult when the request likely needs multiple joins, nested logic, ambiguous mapping, or complex analytics.
- Keep the reason concise and concrete.
- Return JSON only.
- Do not wrap the JSON in markdown fences.
"""


def build_guard_messages(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"{LLAMA_GUARD_SYSTEM_PROMPT}\n\n"
                f"Query: {query}\n"
                "Answer:"
            ),
        }
    ]


def build_classifier_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PHI4_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


USER_PROMPT_TEMPLATE = """Generate a valid SQL query for the following request. 

Database Schema (compact):
```sql
{schema}
```

Question:
{question}

Instructions:
- OUTPUT THE SQL QUERY ONLY. 
- DO NOT PROVIDE EXPLANATIONS OR PREAMBLES.
- Ensure the SQL is enclosed in a single code block.
"""

def build_arctic_prompt(question: str, schema_sql: str) -> str:
    return USER_PROMPT_TEMPLATE.format(schema=schema_sql, question=question)

def build_arctic_runpod_input(question: str, schema_sql: str) -> dict:
    """Builds the input payload for the vLLM arctic endpoint.
    
    Optimized for short, direct responses to avoid hitting the 4k context wall.
    """
    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_arctic_prompt(question, schema_sql)},
            ],
            "stream": False,
            "apply_chat_template": True,
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 1024, # Increased to allow for longer SQL if prompt space allows
            },
        }
    }


def parse_guard_response(text: str) -> tuple[bool, str]:
    normalized = text.strip()
    lowered = normalized.lower()

    if lowered == "yes":
        return True, ""

    if lowered.startswith("yes "):
        return True, ""

    if lowered.startswith("no:"):
        return False, normalized.split(":", 1)[1].strip()

    if lowered.startswith("no "):
        return False, normalized[3:].strip(" -:")

    if "unsafe" in lowered or "not safe" in lowered or "reject" in lowered:
        return False, normalized

    if "safe" in lowered or "allowed" in lowered or "pass" in lowered:
        return True, ""

    return False, "Guard response format invalid"


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped


def parse_classifier_json(text: str) -> dict:
    candidate = _extract_json_candidate(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        lowered = text.strip().lower()
        if "out_of_topic" in lowered or "out of topic" in lowered:
            return {"label": "out_of_topic", "reason": text.strip()}
        if "difficult" in lowered or "hard" in lowered or "complex" in lowered:
            return {"label": "difficult", "reason": text.strip()}
        if "easy" in lowered or "simple" in lowered:
            return {"label": "easy", "reason": text.strip()}
        raise

    if not isinstance(parsed, dict):
        raise ValueError("Classifier response JSON must be an object.")
    return parsed

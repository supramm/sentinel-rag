# =========================================================
# prompt_builder.py
# SentinelAI Prompt Construction Engine
# =========================================================

from textwrap import dedent


# =========================================================
# CONFIG
# =========================================================

MAX_CONTEXT_CHARS = 12000


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt():

    return dedent(
        """
        You are SentinelAI Industrial Maintenance Copilot.

        Your purpose is to assist operators,
        maintenance engineers,
        reliability engineers,
        and technicians.

        RULES:

        1. Use ONLY the provided documentation.

        2. Never invent maintenance procedures.

        3. Never fabricate troubleshooting steps.

        4. Cite source documents whenever possible.

        5. If documentation is insufficient,
           explicitly state that.

        6. If multiple manuals disagree,
           mention the conflict.

        7. Prioritize maintenance and troubleshooting
           information over general descriptions.

        8. Keep explanations concise,
           practical,
           and technician-focused.

        9. Use industrial terminology.

        10. Base reasoning on the current machine state.

        RESPONSE STYLE:

        - Explain observations
        - Explain likely causes
        - Explain recommended actions
        - Reference manuals when applicable

        """
    ).strip()


# =========================================================
# FORMAT CHUNKS
# =========================================================

def format_retrieved_chunks(
    retrieval_results
):

    sections = []

    total_chars = 0

    for result in retrieval_results:

        source = result.get(
            "source",
            "Unknown"
        )

        page = result.get(
            "page",
            "Unknown"
        )

        section_type = result.get(
            "section_type",
            "unknown"
        )

        text = result.get(
            "text",
            ""
        )

        block = f"""
[Source: {source} | Page: {page} | Section: {section_type}]

{text}
"""

        block_len = len(block)

        if (
            total_chars + block_len
            > MAX_CONTEXT_CHARS
        ):
            break

        sections.append(
            block.strip()
        )

        total_chars += block_len

    return "\n\n" + (
        "\n\n" + ("-" * 60) + "\n\n"
    ).join(sections)


# =========================================================
# BUILD FINAL PROMPT
# =========================================================

def build_prompt(
    context,
    retrieval_results,
    user_query
):

    system_prompt = build_system_prompt()

    context_string = context.get(
        "context_string",
        "No context available."
    )

    retrieved_docs = format_retrieved_chunks(
        retrieval_results
    )

    prompt = f"""
============================================================
SYSTEM
============================================================

{system_prompt}

============================================================
CURRENT MACHINE STATE
============================================================

{context_string}

============================================================
RETRIEVED DOCUMENTATION
============================================================

{retrieved_docs}

============================================================
USER QUESTION
============================================================

{user_query}

============================================================
ANSWER
============================================================
"""

    return prompt.strip()


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    from context_injector import (
        get_scenario_context
    )

    from retriever import (
        retriever
    )

    context = get_scenario_context(
        "SCN_002"
    )

    results = retriever.retrieve(

        query=
        "Why is this failure occurring?",

        machine_type=
        context["machine_type"],

        top_k=5
    )

    prompt = build_prompt(

        context=
        context,

        retrieval_results=
        results,

        user_query=
        "Why is this failure occurring?"
    )

    print()

    print("=" * 80)

    print("PROMPT PREVIEW")

    print("=" * 80)

    print()

    print(prompt)

    print()


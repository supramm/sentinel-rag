# =========================================================
# hallucination_gate.py
# SentinelAI Retrieval Validation Layer
# =========================================================

from pathlib import Path
from datetime import datetime


# =========================================================
# CONFIG
# =========================================================

MIN_AVG_COSINE = 0.25
MIN_AVG_JACCARD = 0.01
MIN_AVG_FINAL = 0.30

NON_INDUSTRIAL_REJECTION_THRESHOLD = 0.35


INDUSTRIAL_KEYWORDS = {

    # Existing
    "bearing",
    "temperature",
    "pressure",
    "vibration",
    "motor",
    "coolant",
    "hydraulic",
    "spindle",
    "belt",
    "roller",
    "welding",
    "servo",
    "alarm",
    "fault",
    "failure",
    "maintenance",
    "repair",
    "lubrication",
    "pump",
    "cylinder",

    # Added
    "robotic",
    "robot",
    "arm",
    "calibration",
    "alignment",
    "conveyor",
    "cnc",
    "axis",
    "weld",
    "tooling",
    "fixture",
    "machine",
    "equipment",
    "overheating",
    "inspection",
    "troubleshooting",
    "degradation",
    "actuator",
    "gearbox",
    "sensor",
    "encoder",
    "drive"
}

# =========================================================
# NON INDUSTRIAL TERMS
# =========================================================

NON_INDUSTRIAL_TERMS = {

    "wifi",
    "password",

    "instagram",
    "facebook",
    "twitter",
    "tiktok",
    "youtube",

    "gmail",
    "email",

    "minecraft",
    "fortnite",

    "spotify",
    "netflix",

    "weather",

    "iphone",
    "android",
    "macbook",
    "windows"
}
LOG_FILE = Path(
    "refused_queries.log"
)


# =========================================================
# LOG REFUSALS
# =========================================================

def log_refusal(
    query,
    reason
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"[{timestamp}] "
            f"{query} | "
            f"{reason}\n"
        )


# =========================================================
# INDUSTRIAL QUERY CHECK
# =========================================================

def looks_industrial(
    query
):

    query = query.lower()

    return any(

        keyword in query

        for keyword in INDUSTRIAL_KEYWORDS
    )

# =========================================================
# NON INDUSTRIAL CHECK
# =========================================================

def contains_non_industrial_terms(
    query
):

    query = query.lower()

    return any(

        term in query

        for term in NON_INDUSTRIAL_TERMS
    )
# =========================================================
# MAIN GATE
# =========================================================

def evaluate_retrieval(
    query,
    results
):

    # -----------------------------------------------------
    # RULE 1
    # -----------------------------------------------------

    if len(results) == 0:

        reason = (
            "No relevant documents retrieved."
        )

        log_refusal(
            query,
            reason
        )

        return {

            "passed": False,

            "reason": reason
        }

    # -----------------------------------------------------
    # AGGREGATE SCORES
    # -----------------------------------------------------

    avg_cosine = sum(

        r["cosine_score"]

        for r in results

    ) / len(results)

    avg_jaccard = sum(

        r["jaccard_score"]

        for r in results

    ) / len(results)

    avg_final = sum(

        r["final_score"]

        for r in results

    ) / len(results)

    # -----------------------------------------------------
    # COSINE CHECK
    # -----------------------------------------------------

    if avg_cosine < MIN_AVG_COSINE:

        reason = (
            f"Average cosine too low "
            f"({avg_cosine:.3f})"
        )

        log_refusal(
            query,
            reason
        )

        return {

            "passed": False,

            "reason": reason
        }

    # -----------------------------------------------------
    # JACCARD CHECK
    # -----------------------------------------------------

    if avg_jaccard < MIN_AVG_JACCARD:

        reason = (
            f"Average jaccard too low "
            f"({avg_jaccard:.3f})"
        )

        log_refusal(
            query,
            reason
        )

        return {

            "passed": False,

            "reason": reason
        }

    # -----------------------------------------------------
    # FINAL SCORE CHECK
    # -----------------------------------------------------

    if avg_final < MIN_AVG_FINAL:

        reason = (
            f"Average final score too low "
            f"({avg_final:.3f})"
        )

        log_refusal(
            query,
            reason
        )

        return {

            "passed": False,

            "reason": reason
        }

    # -----------------------------------------------------
    # INDUSTRIAL RELEVANCE CHECK
    #
    # Only reject if:
    # 1. Query doesn't look industrial
    # 2. Retrieval confidence is weak
    # -----------------------------------------------------

    if (
        not looks_industrial(query)
        and avg_final < NON_INDUSTRIAL_REJECTION_THRESHOLD
    ):

        reason = (
            "Query does not appear "
            "to be industrial maintenance related."
        )

        log_refusal(
            query,
            reason
        )

        return {

            "passed": False,

            "reason": reason
        }
    # -----------------------------------------------------
    # NON-INDUSTRIAL HARD REJECTION
    # -----------------------------------------------------

    if contains_non_industrial_terms(query):

        reason = (
            "Query appears unrelated to "
            "industrial maintenance."
        )

        log_refusal(
            query,
            reason
        )

        return {

            "passed": False,

            "reason": reason
        }
    # -----------------------------------------------------
    # PASS
    # -----------------------------------------------------

    return {

        "passed": True,

        "reason": "",

        "avg_cosine": round(
            avg_cosine,
            4
        ),

        "avg_jaccard": round(
            avg_jaccard,
            4
        ),

        "avg_final": round(
            avg_final,
            4
        )
    }
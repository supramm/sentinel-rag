# =========================================================
# context_injector.py
# SentinelAI Runtime Context Builder
# =========================================================

from pathlib import Path
import json


# =========================================================
# PATHS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "Data"

SCENARIOS_PATH = DATA_DIR / "scenarios.json"

RESULTS_PATH = DATA_DIR / "results.json"


# =========================================================
# MACHINE TYPE MAPPING
# Frontend → Retriever
# =========================================================

MACHINE_MAP = {

    "conveyor_belt":
        "conveyor_system",

    "robotic_arm":
        "robotic_welding_arm",

    "cnc_machine":
        "cnc_machine",

    "hydraulic_press":
        "hydraulic_press"
}


# =========================================================
# LOAD JSON FILE
# =========================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# LOAD SCENARIOS
# =========================================================

def load_scenarios():

    raw = load_json(
        SCENARIOS_PATH
    )

    if isinstance(raw, dict) and "scenarios" in raw:
        return raw["scenarios"]

    return raw


# =========================================================
# LOAD RESULTS
# =========================================================

def load_results():

    return load_json(
        RESULTS_PATH
    )


# =========================================================
# INDEX RESULTS BY SCENARIO ID
# =========================================================

def build_results_lookup(results):

    lookup = {}

    for item in results:

        scenario_id = item["scenario_id"]

        lookup[scenario_id] = item

    return lookup


# =========================================================
# EXTRACT TOP FEATURES
# =========================================================

def extract_top_features(result):

    features = []

    lstm_features = result.get(
        "lstm_ae_top3_features",
        []
    )

    if_features = result.get(
        "if_top3_features",
        []
    )

    features.extend(lstm_features)

    features.extend(if_features)

    unique_features = []

    for feature in features:

        if feature not in unique_features:

            unique_features.append(feature)

    return unique_features[:5]


# =========================================================
# EXTRACT SENSOR SNAPSHOT
# =========================================================

def extract_sensor_snapshot(scenario):

    display = scenario.get(
        "display",
        {}
    )

    snapshot = display.get(
        "sensor_snapshot",
        {}
    )

    return snapshot


# =========================================================
# BUILD CONTEXT STRING
# =========================================================

def build_context_string(context):

    lines = []

    lines.append(
        f"SCENARIO ID: {context['scenario_id']}"
    )

    lines.append("")

    lines.append(
        f"MACHINE TYPE: {context['machine_type']}"
    )

    lines.append(
        f"ANOMALY TYPE: {context['anomaly_type']}"
    )

    lines.append(
        f"HEALTH STATUS: {context['health_status']}"
    )

    lines.append(
        f"SEVERITY: {context['severity']}"
    )

    lines.append(
        f"REMAINING USEFUL LIFE: {context['rul_days']} days"
    )

    lines.append("")

    lines.append(
        "TOP CONTRIBUTING FEATURES:"
    )

    for feature in context["top_features"]:

        lines.append(
            f"- {feature}"
        )

    lines.append("")

    lines.append(
        "CURRENT SENSOR SNAPSHOT:"
    )

    for sensor, value in context["sensor_snapshot"].items():

        lines.append(
            f"{sensor}: {value}"
        )

    lines.append("")

    lines.append(
        "MAINTENANCE MESSAGE:"
    )

    lines.append(
        context["maintenance_message"]
    )

    return "\n".join(lines)


# =========================================================
# MAIN CONTEXT BUILDER
# =========================================================

def get_scenario_context(
    scenario_id
):

    scenarios = load_scenarios()

    results = load_results()

    results_lookup = build_results_lookup(
        results
    )

    # -----------------------------------------------------
    # FIND SCENARIO
    # -----------------------------------------------------

    scenario = None

    for item in scenarios:

        if item["scenario_id"] == scenario_id:

            scenario = item

            break

    if scenario is None:

        raise ValueError(
            f"Scenario not found: {scenario_id}"
        )

    # -----------------------------------------------------
    # FIND RESULT
    # -----------------------------------------------------

    result = results_lookup.get(
        scenario_id
    )

    if result is None:

        raise ValueError(
            f"Result not found: {scenario_id}"
        )

    # -----------------------------------------------------
    # MACHINE TYPE
    # -----------------------------------------------------

    frontend_machine_type = scenario[
        "machine_type"
    ]

    machine_type = MACHINE_MAP.get(
        frontend_machine_type,
        frontend_machine_type
    )

    # -----------------------------------------------------
    # HEALTH STATUS
    # -----------------------------------------------------

    health_status = result.get(
        "ground_truth",
        {}
    ).get(
        "health_status",
        "unknown"
    )

    # -----------------------------------------------------
    # RUL
    # -----------------------------------------------------

    rul_days = result.get(
        "ground_truth",
        {}
    ).get(
        "rul_days",
        "unknown"
    )

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = {

        "scenario_id":
            scenario_id,

        "machine_type":
            machine_type,

        "frontend_machine_type":
            frontend_machine_type,

        "anomaly_type":
            scenario.get(
                "anomaly_type",
                "unknown"
            ),

        "severity":
            scenario.get(
                "severity",
                "unknown"
            ),

        "health_status":
            health_status,

        "rul_days":
            rul_days,

        "top_features":
            extract_top_features(
                result
            ),

        "sensor_snapshot":
            extract_sensor_snapshot(
                scenario
            ),

        "maintenance_message":
            result.get(
                "maintenance_message",
                ""
            ),

        "model_consensus":
            result.get(
                "model_consensus",
                False
            ),

        "ensemble_anomaly_score":
            result.get(
                "ensemble_anomaly_score",
                None
            )
    }

    context[
        "context_string"
    ] = build_context_string(
        context
    )

    return context


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    context = get_scenario_context(
        "SCN_002"
    )

    print()

    print("=" * 60)

    print("RUNTIME CONTEXT")

    print("=" * 60)

    print()

    print(
        context["context_string"]
    )

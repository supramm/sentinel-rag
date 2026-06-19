"""
SentinelAI Investigation Assistant.

Standalone Streamlit surface for industrial RAG investigations.
"""

from __future__ import annotations

import html
import json
import random
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "Data"
UI_SCENARIOS_PATH = DATA_DIR / "ui_scenarios.json"
UI_RESULTS_PATH = DATA_DIR / "ui_results.json"


SEVERITY_COLORS = {
    "critical": {"fg": "#ff3b5c", "bg": "rgba(255,59,92,0.08)", "border": "rgba(255,59,92,0.34)"},
    "high": {"fg": "#ff6b3d", "bg": "rgba(255,107,61,0.08)", "border": "rgba(255,107,61,0.32)"},
    "medium": {"fg": "#ffb400", "bg": "rgba(255,180,0,0.08)", "border": "rgba(255,180,0,0.30)"},
    "low": {"fg": "#00b4ff", "bg": "rgba(0,180,255,0.08)", "border": "rgba(0,180,255,0.28)"},
    "none": {"fg": "#00ffc8", "bg": "rgba(0,255,200,0.08)", "border": "rgba(0,255,200,0.26)"},
}

MACHINE_BADGES = {
    "conveyor_belt": "CNV",
    "conveyor_system": "CNV",
    "hydraulic_press": "HYD",
    "cnc_machine": "CNC",
    "robotic_welding_arm": "ROB",
    "robotic_arm": "ROB",
}


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def display_feature_name(feature: str) -> str:
    return str(feature).replace("_", " ").title()


def format_metric_value(value: Any, suffix: str = "", percent: bool = False) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, (int, float)):
        if percent:
            return f"{value * 100:.1f}%"
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def severity_palette(severity: str | None) -> dict[str, str]:
    return SEVERITY_COLORS.get(str(severity or "none").lower(), SEVERITY_COLORS["none"])


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_ui_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return load_json(UI_SCENARIOS_PATH), load_json(UI_RESULTS_PATH)


def find_result_for_scenario(
    scenario: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    scenario_id = scenario.get("scenario_id")
    return next((item for item in results if item.get("scenario_id") == scenario_id), {})


def normalize_rag_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        latency = result.get("latency", {})
        return {
            "success": bool(result.get("success")),
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "refused": bool(result.get("refused")),
            "reason": result.get("reason"),
            "error": result.get("error"),
            "backend": result.get("backend", ""),
            "latency_ms": result.get("latency_ms", latency.get("total_ms")),
        }

    latency = getattr(result, "latency", {}) or {}
    return {
        "success": bool(getattr(result, "success", False)),
        "answer": getattr(result, "answer", None),
        "sources": getattr(result, "sources", []) or [],
        "refused": bool(getattr(result, "refused", False)),
        "reason": getattr(result, "reason", None),
        "error": getattr(result, "error", None),
        "backend": getattr(result, "backend", ""),
        "latency_ms": latency.get("total_ms"),
    }


def patch_backend_data_paths() -> None:
    import context_injector

    context_injector.DATA_DIR = DATA_DIR
    context_injector.SCENARIOS_PATH = DATA_DIR / "scenarios.json"
    context_injector.RESULTS_PATH = DATA_DIR / "results.json"


def run_investigation(scenario_id: str, question: str) -> dict[str, Any]:
    patch_backend_data_paths()
    from rag_chain import run

    return normalize_rag_result(
        run(
            scenario_id=scenario_id,
            question=question,
            llm_backend="groq",
        )
    )


def install_theme() -> None:
    st.markdown(
        """
<style>
#MainMenu, footer, header, .stToolbar,
[data-testid="stToolbar"], [data-testid="stHeader"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background: transparent !important;
    padding-top: 0.55rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"] { display: none !important; }
body {
    background: radial-gradient(ellipse at 20% 30%,
        rgba(0,255,200,0.04) 0%,
        rgba(0,80,255,0.03) 40%,
        #030712 100%) !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    color: #e2e8f0 !important;
}
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, div {
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
.sentinel-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0.45rem 0.2rem 0.8rem;
    border-bottom:1px solid rgba(0,255,200,0.08);
    margin-bottom:0.85rem;
}
.header-left { display:flex; align-items:center; gap:0.7rem; }
.live-dot {
    width:8px; height:8px; border-radius:50%;
    background:#00ffc8;
    box-shadow:0 0 10px #00ffc8;
    animation:header-pulse 1.5s ease-in-out infinite;
}
.header-title {
    font-size:0.66rem;
    letter-spacing:0.25em;
    color:rgba(0,255,200,0.78) !important;
    text-transform:uppercase;
}
.header-meta {
    display:flex; gap:1.1rem; flex-wrap:wrap; justify-content:flex-end;
}
.header-meta span {
    font-size:0.56rem;
    color:rgba(255,255,255,0.28) !important;
    letter-spacing:0.08em;
    text-transform:uppercase;
}
.section-label {
    font-size:0.58rem;
    letter-spacing:0.22em;
    color:rgba(0,255,200,0.46) !important;
    text-transform:uppercase;
    margin:0.45rem 0 0.55rem;
    display:flex;
    align-items:center;
    gap:0.5rem;
}
.section-label::before {
    content:'';
    display:inline-block;
    width:20px; height:1px;
    background:rgba(0,255,200,0.42);
}
.mission-card, .metric-block, .answer-card, .source-card, .question-card {
    background:rgba(0,255,200,0.035);
    border:1px solid rgba(0,255,200,0.16);
    border-radius:10px;
    box-shadow:0 0 28px rgba(0,255,200,0.035);
}
.mission-card {
    padding:1.05rem;
    min-height:130px;
}
.machine-row { display:flex; align-items:center; gap:0.85rem; }
.machine-badge {
    width:56px; height:56px;
    border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    border:1px solid rgba(0,255,200,0.30);
    color:#00ffc8 !important;
    background:linear-gradient(135deg, rgba(0,255,200,0.12), rgba(0,120,255,0.10));
    box-shadow:0 0 22px rgba(0,255,200,0.10);
    font-size:1rem;
    font-weight:900;
    letter-spacing:0.08em;
}
.machine-name {
    font-size:1.15rem;
    font-weight:800;
    letter-spacing:0.04em;
    color:#e2e8f0 !important;
}
.machine-sub {
    font-size:0.58rem;
    color:rgba(255,255,255,0.36) !important;
    letter-spacing:0.12em;
    text-transform:uppercase;
    margin-top:0.25rem;
}
.metric-block {
    padding:0.82rem 0.9rem;
    min-height:92px;
}
.metric-label {
    font-size:0.58rem;
    letter-spacing:0.14em;
    color:rgba(0,255,200,0.55) !important;
    text-transform:uppercase;
    margin-bottom:0.35rem;
}
.metric-value {
    font-size:1.05rem;
    font-weight:800;
    color:#00ffc8 !important;
    line-height:1.15;
    overflow-wrap:anywhere;
}
.metric-unit {
    font-size:0.56rem;
    color:rgba(255,255,255,0.34) !important;
    margin-top:0.22rem;
    letter-spacing:0.06em;
    text-transform:uppercase;
}
.pill-row { display:flex; flex-wrap:wrap; gap:0.45rem; }
.stat-pill {
    font-size:0.58rem;
    padding:0.24rem 0.68rem;
    background:rgba(0,255,200,0.06);
    border:1px solid rgba(0,255,200,0.15);
    border-radius:999px;
    color:rgba(0,255,200,0.72) !important;
    letter-spacing:0.07em;
}
.consensus-row {
    display:flex;
    justify-content:space-between;
    gap:1rem;
    padding:0.42rem 0;
    border-bottom:1px solid rgba(0,255,200,0.06);
    font-size:0.66rem;
}
.consensus-row span:first-child {
    color:rgba(0,255,200,0.62) !important;
    letter-spacing:0.10em;
}
.answer-card {
    padding:1rem 1.05rem;
    min-height:230px;
    line-height:1.65;
    color:rgba(226,232,240,0.92) !important;
    font-size:0.82rem;
}
.answer-placeholder {
    height:210px;
    display:flex;
    align-items:center;
    justify-content:center;
    text-align:center;
    color:rgba(255,255,255,0.30) !important;
    letter-spacing:0.12em;
    text-transform:uppercase;
    font-size:0.64rem;
}
.source-card {
    padding:0.75rem 0.85rem;
    margin-bottom:0.55rem;
}
.source-title {
    display:flex;
    justify-content:space-between;
    gap:0.75rem;
    color:#00ffc8 !important;
    font-size:0.66rem;
    font-weight:800;
    letter-spacing:0.08em;
    text-transform:uppercase;
}
.source-text {
    margin-top:0.45rem;
    color:rgba(226,232,240,0.76) !important;
    font-size:0.70rem;
    line-height:1.55;
}
.stButton > button {
    width:100%;
    background:linear-gradient(135deg, rgba(0,255,200,0.12), rgba(0,120,255,0.12)) !important;
    border:1px solid rgba(0,255,200,0.34) !important;
    color:#00ffc8 !important;
    font-family:'JetBrains Mono', monospace !important;
    font-size:0.70rem !important;
    font-weight:800 !important;
    letter-spacing:0.14em !important;
    text-transform:uppercase !important;
    border-radius:10px !important;
    padding:0.68rem 1rem !important;
    transition:all 0.22s ease !important;
    box-shadow:0 0 20px rgba(0,255,200,0.08) !important;
    white-space:normal !important;
    min-height:44px !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg, rgba(0,255,200,0.22), rgba(0,120,255,0.22)) !important;
    box-shadow:0 0 34px rgba(0,255,200,0.20) !important;
    transform:translateY(-1px) !important;
}
.stTextInput input {
    background:rgba(0,255,200,0.035) !important;
    border:1px solid rgba(0,255,200,0.18) !important;
    color:#e2e8f0 !important;
    border-radius:10px !important;
}
.stExpander {
    border:1px solid rgba(0,255,200,0.12) !important;
    border-radius:10px !important;
    background:rgba(0,255,200,0.025) !important;
}
hr {
    border-color:rgba(0,255,200,0.08) !important;
}
@keyframes header-pulse { 0%, 100% { opacity:1; } 50% { opacity:0.3; } }
@media (max-width: 900px) {
    .sentinel-header { align-items:flex-start; flex-direction:column; gap:0.6rem; }
    .header-meta { justify-content:flex-start; }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
<div class="sentinel-header">
  <div class="header-left">
    <div class="live-dot"></div>
    <span class="header-title">SentinelAI Investigation Assistant</span>
  </div>
  <div class="header-meta">
    <span>Industrial RAG</span>
    <span>Scenario Console</span>
    <span>Tata Tech</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: Any, unit: str = "", color: str = "#00ffc8") -> None:
    st.markdown(
        f"""
<div class="metric-block">
  <div class="metric-label">{escape(label)}</div>
  <div class="metric-value" style="color:{color} !important;">{escape(value)}</div>
  <div class="metric-unit">{escape(unit)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_machine_card(scenario: dict[str, Any], result: dict[str, Any]) -> None:
    machine_type = scenario.get("machine_type", "")
    badge = MACHINE_BADGES.get(machine_type, "SYS")
    palette = severity_palette(scenario.get("severity"))
    st.markdown(
        f"""
<div class="mission-card" style="border-color:{palette['border']};">
  <div class="machine-row">
    <div class="machine-badge" style="color:{palette['fg']} !important;border-color:{palette['border']};">
      {escape(badge)}
    </div>
    <div>
      <div class="machine-name">{escape(scenario.get("machine_name", "Unknown Machine"))}</div>
      <div class="machine-sub">{escape(scenario.get("label", ""))}</div>
    </div>
  </div>
  <div style="margin-top:0.85rem;font-size:0.68rem;color:rgba(255,255,255,0.44) !important;line-height:1.5;">
    {escape(result.get("maintenance_message") or scenario.get("description", ""))}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_left_panel(scenario: dict[str, Any], result: dict[str, Any]) -> None:
    palette = severity_palette(scenario.get("severity"))
    render_machine_card(scenario, result)

    st.markdown('<div class="section-label">Key Scenario Signals</div>', unsafe_allow_html=True)
    top_a, top_b = st.columns(2)
    with top_a:
        render_metric("Machine Name", scenario.get("machine_name"), scenario.get("scenario_id", ""))
    with top_b:
        render_metric(
            "Diagnosis",
            display_feature_name(result.get("diagnosis") or scenario.get("anomaly_type", "unknown")),
            "Predicted failure",
            palette["fg"],
        )

    mid_a, mid_b, mid_c = st.columns(3)
    with mid_a:
        render_metric("Severity", str(scenario.get("severity", "unknown")).upper(), "Scenario", palette["fg"])
    with mid_b:
        render_metric("RUL", format_metric_value(result.get("remaining_useful_life_days"), "d"), "Remaining")
    with mid_c:
        render_metric("Confidence", result.get("confidence_score", "N/A"), "Diagnosis")

    st.markdown('<div class="section-label">Top Contributing Features</div>', unsafe_allow_html=True)
    contributors = scenario.get("top_contributors", [])[:5]
    pills = "".join(
        f'<span class="stat-pill">{escape(display_feature_name(feature))}</span>'
        for feature in contributors
    )
    st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Consensus</div>', unsafe_allow_html=True)
    lstm = result.get("model_outputs", {}).get("lstm_ae", {})
    iso = result.get("model_outputs", {}).get("isolation_forest", {})
    rul = result.get("model_outputs", {}).get("rul", {})
    rows = [
        ("LSTM-AE", "ANOMALY" if lstm.get("anomaly_flag") else "NOMINAL"),
        ("IF", "ANOMALY" if iso.get("anomaly_flag") else "NOMINAL"),
        ("RUL", str(rul.get("bracket", "unknown")).upper()),
        ("AGREEMENT", str(result.get("consensus_strength", "unknown")).replace("_", " ").upper()),
        ("ENSEMBLE", format_metric_value(result.get("ensemble_anomaly_score"), percent=True)),
    ]
    st.markdown(
        "".join(
            f'<div class="consensus-row"><span>{escape(label)}</span><span>{escape(value)}</span></div>'
            for label, value in rows
        ),
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.markdown(
            '<div class="source-text">No source evidence returned for this investigation.</div>',
            unsafe_allow_html=True,
        )
        return

    for source in sources[:5]:
        score = source.get("final_score")
        score_text = f"Score {score}" if score is not None else ""
        st.markdown(
            f"""
<div class="source-card">
  <div class="source-title">
    <span>{escape(source.get("source", "Unknown source"))} / Page {escape(source.get("page", "N/A"))}</span>
    <span>{escape(score_text)}</span>
  </div>
  <div class="source-text">{escape(source.get("text", ""))}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_answer(result: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-label">Investigation Findings</div>', unsafe_allow_html=True)
    if not result:
        st.markdown(
            '<div class="answer-card answer-placeholder">Select a question to begin investigation.</div>',
            unsafe_allow_html=True,
        )
        return

    if result.get("success"):
        body = result.get("answer") or "No answer returned."
        meta = []
        if result.get("backend"):
            meta.append(f"Backend: {result['backend']}")
        if result.get("latency_ms") is not None:
            meta.append(f"Latency: {result['latency_ms']:.0f} ms")
        meta_text = " / ".join(meta)
        st.markdown(
            f"""
<div class="answer-card">
  <div style="white-space:pre-wrap;">{escape(body)}</div>
  <div class="metric-unit" style="margin-top:0.85rem;">{escape(meta_text)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    elif result.get("refused"):
        st.markdown(
            f"""
<div class="answer-card" style="border-color:rgba(255,180,0,0.30);">
  Investigation refused by retrieval gate.<br><br>
  {escape(result.get("reason") or "Insufficient evidence.")}
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="answer-card" style="border-color:rgba(255,59,92,0.30);">
  Investigation failed.<br><br>
  {escape(result.get("error") or "Unknown error.")}
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("Source Evidence", expanded=True):
        render_sources(result.get("sources", []))


def investigate_current_question(question: str) -> None:
    scenario = st.session_state.get("scenario")
    if not scenario or not question.strip():
        return

    st.session_state.active_question = question.strip()
    with st.spinner("Running industrial investigation..."):
        st.session_state.answer_result = run_investigation(
            scenario_id=scenario["scenario_id"],
            question=question.strip(),
        )


def render_right_panel(scenario: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Recommended Investigations</div>', unsafe_allow_html=True)
    questions = scenario.get("recommended_questions", [])[:4]
    q_cols = st.columns(2)
    for idx, question in enumerate(questions):
        with q_cols[idx % 2]:
            if st.button(question, key=f"question_{idx}", use_container_width=True):
                investigate_current_question(question)

    st.markdown('<div class="section-label">Ask Your Own Question</div>', unsafe_allow_html=True)
    input_col, button_col = st.columns([3, 1])
    with input_col:
        custom_question = st.text_input(
            "Custom investigation question",
            label_visibility="collapsed",
            placeholder="Ask a focused maintenance question...",
            key="custom_question",
        )
    with button_col:
        if st.button("Investigate", key="custom_investigate", use_container_width=True):
            investigate_current_question(custom_question)

    active_question = st.session_state.get("active_question")
    if active_question:
        st.markdown(
            f'<div class="metric-unit" style="margin:0.45rem 0 0.1rem;">Active question: {escape(active_question)}</div>',
            unsafe_allow_html=True,
        )

    render_answer(st.session_state.get("answer_result"))


def select_random_scenario() -> None:
    scenarios, results = load_ui_data()
    scenario = random.choice(scenarios)
    st.session_state.scenario = scenario
    st.session_state.result = find_result_for_scenario(scenario, results)
    st.session_state.answer_result = None
    st.session_state.active_question = None


def render_console() -> None:
    scenario = st.session_state.get("scenario")
    result = st.session_state.get("result")
    if not scenario or not result:
        st.markdown(
            """
<div class="answer-card answer-placeholder" style="min-height:310px;">
  Press Analyze Scenario to load an industrial investigation console.
</div>
""",
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns([4, 6], gap="large")
    with left:
        render_left_panel(scenario, result)
    with right:
        render_right_panel(scenario)


def main() -> None:
    st.set_page_config(
        page_title="SentinelAI Investigation Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    install_theme()
    render_header()

    if "scenario" not in st.session_state:
        st.session_state.scenario = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "answer_result" not in st.session_state:
        st.session_state.answer_result = None
    if "active_question" not in st.session_state:
        st.session_state.active_question = None

    button_col, status_col = st.columns([1, 3])
    with button_col:
        if st.button("Analyze Scenario", key="analyze_scenario", use_container_width=True):
            select_random_scenario()
    with status_col:
        scenario = st.session_state.get("scenario")
        if scenario:
            st.markdown(
                f"""
<div class="pill-row" style="padding-top:0.35rem;">
  <span class="stat-pill">{escape(scenario.get("scenario_id"))}</span>
  <span class="stat-pill">{escape(scenario.get("machine_name"))}</span>
  <span class="stat-pill">{escape(str(scenario.get("severity", "unknown")).upper())}</span>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    render_console()


if __name__ == "__main__":
    main()

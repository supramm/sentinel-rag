# SentinelAI RAG — Industrial Maintenance Investigation Assistant

<p align="center">
  <a href="https://sentinel-rag-u.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-OPEN%20APP-FF4B4B?style=for-the-badge">
  </a>
</p>

<p align="center">
  <strong>Industrial Retrieval-Augmented Generation (RAG) System for Explainable Predictive Maintenance</strong>
</p>

<p align="center">
  🔗 https://sentinel-rag-u.streamlit.app/
</p>

---

## 📸 Screenshot

<p align="center">
  <img width="1920" height="1080" alt="Screenshot 2026-06-19 185202" src="https://github.com/user-attachments/assets/71a214c4-5210-4dc3-b648-6e1b500640c2" />

</p>

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Groq-Llama%203.3%2070B-00A67E?style=for-the-badge">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge">
  <img src="https://img.shields.io/badge/SentenceTransformers-Embeddings-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</p>

---

## 📖 Overview

SentinelAI RAG bridges the gap between predictive maintenance outputs and actionable engineering insights.

The system combines machine health scenarios, engineering documentation, vector retrieval, and Large Language Models to generate evidence-backed maintenance investigations.

Instead of providing generic AI responses, SentinelAI retrieves relevant information from industrial maintenance manuals and uses that evidence to explain:

- Predicted failure modes
- Maintenance recommendations
- Intervention urgency
- Remaining Useful Life (RUL) implications
- Supporting engineering evidence

### Supported Machine Types

- Conveyor Belt Systems
- CNC Machines
- Hydraulic Presses
- Robotic Welding Arms

---

## ✨ Key Features

### Scenario-Aware Investigations

Every investigation begins from a machine scenario containing:

- Machine type
- Predicted failure mode
- Severity level
- Remaining Useful Life (RUL)
- Contributing sensor signals
- Ensemble model outputs

### Industrial Knowledge Retrieval

Relevant evidence is retrieved using:

- FAISS vector similarity search
- Machine-specific filtering
- Semantic ranking
- Knowledge-grounded retrieval

### Hallucination Protection

A retrieval validation layer prevents unsupported responses through:

- Semantic similarity checks
- Keyword overlap validation
- Industrial relevance filtering
- Evidence quality assessment

### LLM-Powered Explanations

Powered by Groq-hosted Llama 3.3 70B:

- Root cause analysis
- Maintenance recommendations
- Failure explanation
- Urgency assessment
- Evidence-grounded responses

### Source Transparency

Each investigation includes:

- Source manual
- Page number
- Retrieval score
- Supporting text excerpts

---

## 🏗️ System Architecture

Scenario Data
      │
      ▼
Context Injector
      │
      ▼
Retriever (FAISS)
      │
      ▼
Hallucination Gate
      │
      ▼
Prompt Builder
      │
      ▼
Groq LLM
      │
      ▼
Investigation Report
      │
      ▼
Source Evidence


---

## 🔄 Investigation Workflow

1. Load an industrial machine scenario.
2. Review machine health indicators.
3. Select a recommended investigation question or enter a custom query.
4. Retrieve supporting engineering evidence.
5. Generate an evidence-backed investigation report.
6. Review source references and maintenance documentation.

---

## 💡 Example Questions

- What maintenance action is recommended for this machine?
- How urgent is intervention given the predicted RUL?
- Does the evidence support the predicted failure mode?
- Which sensor signals contributed most to this alert?
- What failure mechanism best matches the observed symptoms?

---

## 🤖 Supported Machines

| Machine | Example Failure Modes |
|----------|----------------------|
| Conveyor Belt | Thermal Runaway, Belt Wear, Motor Failure |
| CNC Machine | Bearing Degradation, Tool Wear |
| Hydraulic Press | Pressure Loss, Seal Failure |
| Robotic Welding Arm | Joint Degradation, Servo Failure |

---

## 🚀 Live Demo

### Streamlit Deployment

https://sentinel-rag-u.streamlit.app/

---

## 📂 Project Structure

rag_clean/
│
├── rag_app.py
├── rag_chain.py
├── llm_client.py
├── retriever.py
├── context_injector.py
├── hallucination_gate.py
├── prompt_builder.py
│
├── Data/
│   ├── scenarios.json
│   ├── results.json
│   ├── ui_scenarios.json
│   └── ui_results.json
│
├── kb/
│   ├── index.faiss
│   └── metadata.json
│
├── requirements.txt
└── .gitignore
---

## ⚙️ Local Setup

Clone the repository:

```bash
git clone <repository-url>
cd rag_clean
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run the application:

```bash
streamlit run rag_app.py
```

---

## 🔮 Future Roadmap

### V2

- Scenario-driven industrial investigations
- Explainable maintenance recommendations
- Evidence-grounded RAG pipeline

### V3

- React dashboard integration
- Embedded conversational assistant
- Real-time machine investigations
- Multi-turn diagnostic workflows

---

## ⚠️ Disclaimer

This project was developed for research, educational, and demonstration purposes. Generated recommendations should be reviewed by qualified maintenance engineers before operational deployment in industrial environments.

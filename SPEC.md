# SPEC.md: AgentDataset Architecture & Plan

## 1. Vision & Goals
- **Product:** AgentDataset (formerly AutoSource).
- **Mission:** An autonomous platform that discovers research documents (web/PDF) and orchestrates specialized AI agents to generate high-fidelity, train-ready synthetic datasets.
- **Key Shift:** Moving from static, linear scripts to a dynamic, modular, and state-machine-driven architecture deployed as a Streamlit Cloud application.

## 2. Core Pillars

### 2.1 The Discovery Layer (Phase 0)
- **Role:** Autonomous knowledge acquisition.
- **Capabilities:** Execute natural language queries to find, filter, and fetch relevant PDFs and HTML content.
- **Integration:** Acts as the primary feed for the Extraction layer.

### 2.2 The Extraction Layer (Phase 1)
- **Role:** Convert raw text into structured statistical "DNA" (`parameters.json`).
- **Capabilities:** 
    - Pluggable extraction schemas (e.g., Survey, Finance).
    - **Pre-flight Checks:** Assess "Statistical Density" to prevent hallucination from narrative-heavy texts.

### 2.3 The Persona Engine & Validation Loop (Phase 2 & 3)
- **Role:** Iterative refinement of synthetic data.
- **The Orchestrator:** Manages the optimization loop, deploying specialized "Personas."
- **Personas:** Distinct modules (e.g., "The Statistician", "The Critic") that propose parametric updates.
- **Validation:** Provides a structured `FidelityReport` (KS-test, Correlation, Bias) to guide the next iteration.

### 2.4 The Dashboard (UI)
- **Role:** The "Glass Box" interface.
- **Stack:** Streamlit.
- **Features:** Unified view of the Discovery, Synthesis heartbeat, live fidelity scoring, and an interactive DATACARD.

## 3. Architecture & State Management

### 3.1 Stateless & Cloud-Ready
- **No Git:** Optimization uses in-memory parametric checkpoints, eliminating dependency on git branches/commits.
- **Session Management:** Unique session IDs (`sessions/{timestamp}/`) track intermediate artifacts and metadata.

### 3.2 Directory Structure
```text
agentdataset/
├── app.py                 # Streamlit main entry point
├── core/
│   ├── orchestrator.py    # State machine and loop control
│   ├── discovery.py       # Web search and document fetch
│   ├── extractor.py       # PDF/HTML parsing & parameter extraction
│   ├── synthesizer.py     # Persona-driven generation logic
│   └── validator.py       # Fidelity scoring
├── models/
│   └── schemas.py         # Pydantic data models for structured IO
└── sessions/              # Ephemeral workspace for active runs
```

## 4. Safety & Stability (Guardrails)

### 4.1 "Caveman" Communication Protocol (Anti-Chattiness)
- **Principle:** Enforce ultra-compressed, filler-free communication across all internal agent interactions to reduce token burn (up to 75%) and prevent LLM "babble".
- **Rules:**
  - **No Fluff:** Drop articles (a, an, the) and filler words (just, really, basically).
  - **No Pleasantries:** Eliminate greetings, hedging, and non-committal language.
  - **Structural Pattern:** Force interactions into strict `[thing] [action] [reason]. [next step].` formats (e.g., "High bias age. Copula skewed. Fix: Increase noise 0.1.").
  - **Exception:** Final outputs (like DATACARD.md and user-facing UI text) revert to professional, clear language.

### 4.2 Cost Control (Token Burn)
- Hard limits on the maximum number of loop iterations and API token usage per session.
- System prompt instructions enforce the "Caveman" protocol for all underlying LLM calls.

### 4.3 UI Concurrency
- Utilization of Streamlit's `st.session_state` and background execution strategies to prevent loop restarts on UI interactions.

### 4.4 Convergence Control (Anti-Plateau)
- Implementation of a "Patience Counter": If fidelity fails to improve after $N$ iterations, force a strategic pivot or gracefully halt.

### 4.5 Data Persistence
- **Auto-Download:** Automatically prompt the user to download the generated `data.csv` and `DATACARD.md` upon successful completion to mitigate Streamlit Cloud's ephemeral filesystem risks.

## 5. Implementation Phases
1. **Phase 1: Project Restructure & Models:** Move scripts into the `core/` package and define Pydantic schemas. Apply Caveman prompt rules.
2. **Phase 2: Discovery & Extraction:** Implement the search agent and robust extraction logic.
3. **Phase 3: The Engine:** Build the Orchestrator, Persona Engine, and Stateless Validation loop.
4. **Phase 4: Streamlit UI:** Wire the backend to `app.py` and implement session tracking and auto-download.

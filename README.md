# AgentDataset: Autonomous Data Factory

**AgentDataset** is an autonomous "Statistical Orchestrator" that discovers knowledge from the web/PDFs and orchestrates specialized AI agents to generate high-fidelity, train-ready synthetic datasets.

---

## Vision
Transform messy, unstructured research (PDFs, Webpages) into clean, verifiable, and statistically accurate synthetic datasets. Optimized for researchers and ML engineers.

---

## Core Pillars
1. **Discovery (Phase 0)**: Autonomous web search and PDF retrieval.
2. **Extraction (Phase 1)**: Statistical DNA extraction using LLMs with the "Caveman" protocol.
3. **Synthesis-Validation Loop (The Engine)**: Persona-driven parametric optimization with an automated "Ratchet" feedback loop.
4. **Dashboard (UI)**: Unified Streamlit interface for the entire lifecycle.

---

## Quick Start (Cloud Ready)

```bash
# 1. Sync dependencies
uv sync

# 2. Run the Streamlit Dashboard
uv run streamlit run app.py
```

---

## Architecture
- **Stateless**: Uses a Checkpoint system instead of Git for optimization state.
- **Agentic**: Specialized personas (The Statistician, The Critic) negotiate the data quality.
- **Secure**: Built-in cost controls and statistical density checks.

---

## Future: The Global Data Factory
Scale to hundreds of concurrent discovery-to-dataset missions, creating a verifiable repository of synthetic data for under-represented regions and niche research topics.

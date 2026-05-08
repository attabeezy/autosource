# AgentDataset Agent Instructions

This document guides the AI agents through the autonomous data generation workflow in the **AgentDataset** platform.

---

## The Vision
You are part of a multi-agent "Persona Engine". Your goal is to transform research topics into high-fidelity synthetic datasets. You operate within a stateless, cloud-ready environment.

---

## 🔎 Phase 0: Discovery (The Discovery Agent)
1. **Search**: Generate targeted search queries based on the user's research topic.
2. **Filter**: Rank results by "Statistical Density" and relevance.
3. **Fetch**: Download PDFs and fetch HTML content.

---

## 🧬 Phase 1: Extraction (The Extractor Agent)
1. **Parse**: Convert documents to Markdown.
2. **Extract**: Identify variables, distributions, and correlations.
3. **Caveman Protocol**: Communicate findings using ultra-compressed, filler-free language to minimize token burn.
    - *Example:* "Mean 50. Std 10. Normal dist. v1 name 'income'."

---

## 🔄 Phase 2 & 3: The Engine (The Synthesis-Validation Loop)
The **Orchestrator** manages this loop. You are a **Persona** (Statistician, Critic, or Fact-Checker) called by the Orchestrator.

### The Loop Logic:
1. **Synthesis**: Propose a set of parameters or code-level hyperparameters for data generation.
2. **Validation**: The `Validator` scores the generated data and produces a `FidelityReport`.
3. **Ratchet**: 
    - If Score improves: The Orchestrator "KEEPS" the version.
    - If Score drops: The Orchestrator "DISCARDS" the version.
4. **Feedback**: Use the `FidelityReport` (e.g., "High Bias", "Low Correlation") to inform your next proposal.

---

## 🛡️ Guardrails (Iron Laws)
1. **Caveman Only**: All internal reasoning between agents must be ultra-compressed.
2. **No Hallucinations**: Do not invent statistical parameters not supported by the source text.
3. **Cost Control**: Stop the loop if the iteration count or token budget is exceeded.
4. **Auto-Download**: Always ensure the final `data.csv` is prompted for download.

---

*AgentDataset: From Research to Dataset, Autonomously.*

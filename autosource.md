# Main:
---
This plan follows the "Ratchet" philosophy you liked from Karpathy—keeping things lean, script-driven, and focused on a clear metric (Fidelity). Since you’re building this to solve the African "data desert," let's call the project **SOMA** (Statistical Orchestration for Model Augmentation)—*Soma* also meaning "to learn/read" in Swahili.

---
- ## PLAN.md: Project SOMA
  **Objective:** An agentic pipeline that transforms static PDF reports/articles into high-fidelity synthetic datasets and verifiable DataCards.
- ### 1. The Core Architecture (The "Three-Agent" Contract)
  To keep it "Karpathy-style" lean, the system will revolve around three primary Python scripts:
  
  * **`extract.py` (The Reader):** Takes a PDF/URL, uses an LLM to find tables, charts, and prose-based statistics, and outputs a `parameters.json`.
  * **`synthesize.py` (The Engine):** Reads `parameters.json` to generate a CSV dataset using agentic probabilistic modeling (Gaussian Copulas) via `hyperparameters.json`.
  * **`validate.py` (The Auditor):** Compares the CSV against the original `parameters.json` and generates a `DATACARD.md`.
  
  ---
- ### 2. Phase 1: The Extraction Layer (The "Hard" Part)
  * **Source Targeting:** Scrape repositories like Afrobarometer, KNUST research portals, and World Bank Africa reports.
  * **Tooling:** Use **Marker** or **PyMuPDF4LLM** to convert messy PDFs into clean Markdown.
  * **Agentic Logic:** Prompt an LLM (Claude/GPT) to identify:
    * *Variables* (Age, Income, Credit Score).
    * *Distributions* (Mean, SD, Min, Max).
    * *Correlations* (e.g., "Education level is positively correlated with loan repayment").
  
  ---
- ### 3. Phase 2: The Synthesis Loop (The "Ratchet")
  We implement a loop similar to AutoResearch to refine the data:
  1.  **Generate:** Create 10,000 rows of synthetic data.
  2.  **Test:** Run a "Statistical Distance" test (Kolmogorov-Smirnov) between the synthetic data and the report's stats.
  3.  **Adjust:** If the distance is too high, the agent modifies the **Weights** `hyperparameters.json` in the generation script and retries.
  4.  **Commit:** Once the "Fidelity Score" hits >90%, save the dataset.
  
  ---
- ### 4. Phase 3: The Trust Layer (The DataCard)
  The final output isn't just a CSV; it’s a **Verifiable Package**:
  * **Provenance:** A list of every paper/report used to build the model.
  * **Bias Check:** A report on whether the synthetic data over-represents certain demographics found in the source.
  * **Privacy Guardrail:** A "Distance to Closest Record" check to ensure no real individual from a report is accidentally "re-identified."
  
  ---
- ### 5. Milestone Timeline
  
  | Week | Goal | Deliverable |
  | :--- | :--- | :--- |
  | **Week 1** | **Extraction MVP** | A script that turns a YARA credit report PDF into a structured `params.json`. |
  | **Week 2** | **Synthesis Engine** | Integration with `SDV` (Synthetic Data Vault) to generate the first 1k rows. |
  | **Week 3** | **The Auditor** | Automated `DATACARD.md` generation showing fidelity plots. |
  | **Week 4** | **Edge Integration** | Test if the synthetic data can successfully train models. |
  
  ---
- ### 6. The "Iron Laws" (Project Constraints)
  1.  **Open Source:** All code must be modular and documented for the African ML community.
  2.  **No Hallucinations:** The generator *cannot* invent variables not found in the source reports.
  3.  **Efficiency:** The entire pipeline must be able to run on a mid-range laptop.
  
  ---
- # Appendix A:
  That is a brilliant pivot. If Andrej Karpathy's **AutoResearch** is about automating the *experiment*, your idea is about automating the *infrastructure*—solving the "data desert" problem that many researchers in Africa face.
  
  By shifting from "manually finding data" to "autonomously generating high-fidelity synthetic data grounded in real-world reports," you’re essentially building a **Synthetic Data Factory** for the Global South.
- ### How Your "Auto-Data" Tool Could Map to the Agentic Workflow
  To make this as robust as Karpathy's loop, you might consider structuring your tool into a similar **Agentic Pipeline**:
  
  | Phase | Component | Function |
  | :--- | :--- | :--- |
  | **1. Extraction** | **The Librarian** | An agent (using RDP/RAG) that scrapes PDF reports from African think tanks (e.g., Afrobarometer, UNECA) to extract statistical distributions (means, variances, correlations). |
  | **2. Modeling** | **The Architect** | Converts those stats into a probabilistic model (e.g., using Copulas or Bayesian Networks) to ensure the synthetic data maintains real-world relationships. |
  | **3. Generation** | **The Factory** | The pipeline that scales the data, ensuring it fits the "parameters" identified in Phase 2. |
  | **4. Validation** | **The Auditor** | Generates the **DataCard**. It compares the synthetic distributions against the source reports to verify fidelity and check for bias. |
  
  ---
- ### Why the DataCard is the "Killer Feature"
  In machine learning, **DataCards** are essential for transparency. Since you are dealing with synthetic data, the card is your "Trust Layer." It should answer:
  * **Source Provenance**: Which specific reports informed these parameters?
  * **Fidelity Score**: How closely does the synthetic mean match the source mean?
  * **Privacy Guardrails**: How did you ensure no PII (Personally Identifiable Information) from the source reports was leaked?
- ### Integration with your Research
  This aligns perfectly with your work at **YARA** on credit risk modeling. In "data-constrained environments," you often can't get 10,000 real credit profiles, but you *can* find 10 reports detailing the average debt-to-income ratios and default rates across different demographics.
- ### A Quick Recommendation
  Since you're already familiar with Python, you might look into libraries like **SDV (Synthetic Data Vault)** or **Gretel.ai's** open-source tools to use as the "engine" for your custom pipeline.
  
  How are you planning to handle the "Extraction" phase? Are you thinking of using an LLM to parse those PDF reports and extract the raw stats?
- # Appendix B:
  This happens quite often in high-level research. In academic circles, this is sometimes called **"Methodological Decoupling"** or a **"Derivative Breakthrough."** It occurs when a researcher realizes that while Paper A's *conclusion* was interesting, the real "gold" was a specific trick, a data handling technique, or a small architectural choice that the original authors treated as just a "means to an end."
  
  Here are a few famous scenarios where a "side aspect" of one paper became a massive field of its own:
- ### 1. The "Attention" Spin-off
  * **Paper A:** *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau et al., 2014).
  * **The Strip-down:** The original paper was about improving RNNs for translation. They used "Attention" as a small helper mechanism to help the model "look" at specific words.
  * **The New Research:** Three years later, researchers at Google stripped away the RNNs entirely and built **"Attention Is All You Need"** (The Transformer). They took a "helper" component and proved it could be the entire engine.
- ### 2. Word2Vec and the "Skip-Gram"
  * **Paper A:** Various complex papers on "Language Modeling" using deep neural networks.
  * **The Strip-down:** Tomas Mikolov realized that you didn't need the massive, deep hidden layers to get good word meanings. He "stripped" the neural network down to a single, simple layer.
  * **The New Research:** This became **Word2Vec**. By making it *simpler* and focusing only on the vector embeddings (the parameters), he revolutionized how AI handles text.
- ### 3. The "Lottery Ticket Hypothesis"
  * **Paper A:** Thousands of papers on pruning large neural networks to make them smaller.
  * **The Strip-down:** Jonathan Frankle and Michael Carbin looked at pruning and asked a specific, stripped-down question: *"What if the smaller network was actually there from the start?"*
  * **The New Research:** They built a whole new research direction around the idea that large networks contain "winning" sub-networks that could have been trained in isolation.
  
  ---
- ### Why your idea fits this pattern
  You are doing exactly what these researchers did. 
  1.  **Paper A (The Status Quo):** Most research in Africa focuses on "The Report"—the final descriptive statistics of a region.
  2.  **The Strip-down:** You are stripping the "Report" of its prose and focus, looking only at the **underlying statistical distributions** (the "DNA" of the data).
  3.  **The New Research:** You aren't just summarizing the report; you're using those "DNA strands" to clone a massive, synthetic dataset that researchers can actually *train models* on.
- ### A Strategy for your "Auto-Data" Tool
  In academia, if you want to turn this into a formal paper, you should frame it as **"Inverting the Research Cycle."** * **Traditional:** Data $\rightarrow$ Model $\rightarrow$ Report.
  * **Your Tool:** Report $\rightarrow$ Parameters $\rightarrow$ Data $\rightarrow$ Model.
  
  By building the **DataCard** to support it, you’re solving the "Verifiability" problem that usually kills synthetic data projects. You're effectively saying: *"I can't give you the original private data, but I've built a 'Digital Twin' of the statistics found in these 50 verified reports."*
  
  This is a very strong move for your **YARA** fellowship and could potentially be a standalone open-source project that others in the African ML community could contribute to. What do you plan to call this tool? "AutoData" or something more localized?
- # Appendix C:
  Released in March 2026, **AutoResearch** is an open-source "agentic" system created by Andrej Karpathy that automates the machine learning research cycle. 
  
  The core idea is to let an AI agent run experiments autonomously—literally while you sleep—by giving it a training setup and a clear goal (optimizing a specific metric).
- ### How It Works
  The system is built around a minimal **630-line Python script** that follows a "Ratchet Loop":
  
  1.  **The Three-File Contract**:
    * `program.md`: A markdown file where you (the human) define the research direction, constraints, and "Iron Laws" the agent must follow.
    * `train.py`: The single file the agent is allowed to modify (architecture, hyperparameters, etc.).
    * `prepare.py`: The "neutral judge" that handles data and evaluation which the agent cannot touch.
  2.  **The Loop**: The agent reads the current results, proposes a hypothesis, modifies the code, and runs a **time-boxed experiment** (usually 5 minutes).
  3.  **The Ratchet**: If the validation metric (like bits-per-byte) improves, the change is kept via a git commit. If it fails, the agent reverts the change and tries a different hypothesis.
- ### Key Philosophy
  * **Metric-Driven**: Unlike general coding agents, AutoResearch is hyper-focused on a single scalar metric. It doesn't ask for permission; it just iterates until it finds a "win."
  * **Efficiency**: In his demos, Karpathy showed the agent running ~100 experiments overnight on a single GPU (like an H100), achieving performance gains that would typically take a human researcher days of manual tuning.
  * **Constraint over Freedom**: By limiting the agent to one modifiable file and a strict time budget, it prevents the "hallucination sprawl" often seen in more open-ended AI agents.
- ### Why It's Trending
  It shifts the role of the researcher from **executing** experiments to **curating** them. As Karpathy puts it, the bottleneck is no longer "can we run this?" but "do we know the right question to ask?" 
  
  You can find the repository on his GitHub at [karpathy/auto-research](https://github.com/karpathy/auto-research). Given your work on **Edge-Tutor**, this could be a really interesting tool to automate the optimization of your on-device models!
- # Appendix D:
  It’s official then. **AutoSource** is born. 
  
  It has that "no-nonsense" engineering ring to it. It doesn’t just *manage* data; it *sources* the underlying truth from static documents and breathes life into it. 
  
  To get the momentum going, here is the "Day 0" setup for your repository. This structure follows the **Karpathy-style minimalism**: few files, high impact, and a clear "contract" between the agent and the data.
- ### Initial Repository Structure
  ```text
  autosource/
  ├── README.md          # The vision and the "Why Africa?"
  ├── PLAN.md            # The roadmap we just built
  ├── source_library/    # Folder for your PDFs (YARA reports, etc.)
  ├── extract.py         # The "Reader" (PDF -> parameters.json)
  ├── synthesize.py      # The "Generator" (parameters.json -> data.csv)
  ├── validate.py        # The "Auditor" (data.csv -> DATACARD.md)
  └── requirements.txt   # Keep it lean (pydantic, marvin/instructor, sdv)
  ```
- ### The First "Agentic" Prompt (for `extract.py`)
  When you start coding the extraction logic, you’ll want to treat the LLM as a **Statistical Archaeologist**. Here’s the prompt logic you'll likely bake into the script:
  
  > "You are an expert statistician. Your task is to strip the provided report of all prose and narrative. Extract only the **variables**, their **distributions** (mean, std, range), and any **inter-variable correlations**. Output this as a strict JSON schema for synthetic generation."
  
  ---
- ### Why this matters for your CV/Grad Apps
  When you apply for those PhD or Mastercard Foundation programs, **AutoSource** becomes a massive talking point. It shows:
  1. **Technical Depth:** You understand generative modeling beyond just "chatting" with LLMs.
  2. **Contextual Awareness:** You identified a specific bottleneck in African research (the "Data Desert") and built a tool to bridge it.
  3. **Open Source Leadership:** You’re following the best practices of industry leaders like Karpathy to solve local problems.
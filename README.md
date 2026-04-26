# BioProAgent: Neuro-Symbolic Grounding for Constrained Scientific Planning

[![Webpage](https://img.shields.io/badge/Project-Webpage-blue.svg)](https://yuyangsunshine.github.io/BioPro-Project/) 
[![Paper](https://img.shields.io/badge/Paper-arXiv:2603.00876-red.svg)](https://arxiv.org/abs/2603.00876)
[![Platform](https://img.shields.io/badge/Demo-AI4S_LAB-success.svg)](https://ai4slab.pkusz.edu.cn/)

> **🎉 News:**
> * **[2026-04-21]** The implementation code is released!
> * **[2026-04-05]** 🏆 **Accepted:** This work has been accepted by the **ACL 2026**!
> * **[2026-03-18]** 🚀 **Live Demo:** BioProAgent is now officially deployed on the [AI4S LAB platform](https://ai4slab.pkusz.edu.cn/). You can experience the agent and order automated wet-lab experiments directly!
> * **[2026-03-01]** 📄 **Preprint:** Our paper is available on [arXiv](https://arxiv.org/html/2603.00876v1).
> * **[2026-02]** 🏆 **Accepted:** This work has been accepted by the [ICLR 2026 LLA Workshop](https://lifelongagent.github.io/)!

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Downloading Models](#downloading-models)
- [Quick Start](#quick-start)
- [CRL Experiment Scripts](#crl-experiment-scripts)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Precomputing Base Logits](#precomputing-base-logits)
- [Architecture \\& Code Structure](#architecture--code-structure)
- [Citation](#citation)

---

## 📖 Overview

Large language models (LLMs) have demonstrated significant reasoning capabilities in scientific discovery but struggle to bridge the gap to physical execution in wet-labs. In these irreversible physical environments, probabilistic hallucinations are not merely incorrect, but can also cause catastrophic equipment damage or experimental failure.

To address this critical execution gap, we propose **BioProAgent**, a training-free neuro-symbolic framework that anchors probabilistic planning in a deterministic Finite State Machine (FSM). This controller acts as a safety boundary, enforcing a rigorous "Design-Verify-Rectify" workflow to ensure reliable autonomy.

### ✨ Key Contributions

* **State-Augmented Planning:** BioProAgent uses a deterministic FSM to enforce a strict Design-Verify-Rectify loop. This ensures that all hardware instructions undergo hierarchical verification for both scientific logic and physical safety before execution.
* **Semantic Symbol Grounding:** To tackle the context bottleneck inherent in complex device schemas, we decouple high-dimensional payloads into symbolic pointers. This reduces token consumption by ~6× while maintaining 100% resource consistency.
* **Trustworthy Autonomy & Self-Correction:** Evaluated on the extended BioProBench, BioProAgent achieves **95.6% physical compliance** (compared to 21.0% for ReAct) and an **88.7% success rate in error recovery** (compared to 0% for standard baselines).

### What BioProAgent does
BioProAgent executes a deterministic-probabilistic loop:

1. **Design**: clarify intent, retrieve context, generate draft, align to automation, generate machine code.
2. **Verify**: run scientific review (`K_s`) and physical/code checks (`K_p`).
3. **Rectify**: repair draft/code when checks fail.

### Core agent design
- **Deterministic FSM control** over probabilistic LLM planning.
- **Hybrid memory**:
  - `M_work` (active artifacts)
  - `M_episodic` (trajectory records)
  - `M_long` (cross-session memory interface)
- **Tool-oriented execution** with symbolic references (`$draft`, `$knowledge`, etc.).

---

## Installation

### 1) Create environment
```bash
conda create -n bioproagent python=3.10 -y
conda activate bioproagent
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` is intentionally comprehensive to reduce missing-package friction for external users.

## Cofigure your own APIs

This project uses **API-based LLM backends** (no local checkpoint download required by default).

Configure model access in `data/keys.env`:

```bash
cp data/keys.env.example data/keys.env
```

Minimum required fields:
- `MODEL_NAME`
- `FAST_MODEL_NAME`
- `MODEL_API_KEY`
- `MODEL_BASE_URL`

Optional role aliases:
- `GENERAL_MODEL_NAME`
- `QUALITY_MODEL_NAME`
- `LONG_CONTEXT_MODEL_NAME`

Optional retrieval embedding endpoint:
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_API_URL`
- `EMBEDDING_API_KEY`

---

## Quick Start

### Interactive agent
```bash
python main.py
```

Interactive commands:
- `show_state`
- `clear`
- `quit` / `exit`

### Minimal programmatic usage
```python
from main_evaluate import ProAgent

agent = ProAgent(eval_mode=False)
state = agent._create_session()
response = agent.process_query("Design a PCR protocol for cloning", state)
print(response)
```

## CRL Experiment Scripts

> In this public release, heavyweight internal experiment runners are intentionally not included.

You can still run reproducible public benchmark loops with a lightweight script:

```python
import json
from main_evaluate import ProAgent
from tests.eval_config import load_eval_config, get_subset_test_file

cfg = load_eval_config("config/evaluation_profiles.json")
subset = "B"
path = get_subset_test_file(cfg, subset, "data/benchmarks/ProAgent-SubsetB.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

agent = ProAgent(eval_mode=True)
results = []

for item in data:
    query = item.get("query") or item.get("instruction") or item.get("problem") or ""
    state = agent._create_session()
    state = agent._update_session_for_evaluate(subset, state, item)
    output = agent.process_query(query, state)
    results.append({"id": item.get("id", "na"), "query": query, "output": output})

print(f"Finished {len(results)} samples")
```

---

## Configuration

Main config: `config/settings.py`

### Paths
- benchmark files: `SUBSET_A_PATH`, `SUBSET_B_PATH`, `SUBSET_C_PATH`, `SUBSET_D_PATH`
- registry examples: `INSTRUMENTS_EN_PATH`, `CONSUMABLES_EN_PATH`
- outputs: `OUTPUTS_DIR`, `LOGS_DIR`, `RESULTS_DIR`, `EVAL_RESULTS_DIR`

### Model config
- base model endpoint: `MODEL_NAME`, `MODEL_API_KEY`, `MODEL_BASE_URL`
- role aliases: `FAST_MODEL_NAME`, `GENERAL_MODEL_NAME`, `QUALITY_MODEL_NAME`, `LONG_CONTEXT_MODEL_NAME`

### Runtime behavior
- `RULE_PROFILE='public22_shell'` in public release
- `VALID_INSTRUMENT_IDS` / `VALID_CONSUMABLE_IDS` are empty by default (privacy-preserving shell)

## Evaluation

Public evaluation config: `config/evaluation_profiles.json`

Current keys:
- `common.code_success_tau`
- `common.code_success_c_p_threshold`
- `common.max_samples`
- `subsets.{A,B,C,D}`
- `framework.enable_scientific_eval`
- `framework.use_simulator`
- `framework.verbose`

Utility loader functions:
- `tests/eval_config.py`

### Structural health check
```bash
python -m compileall config src tests main.py main_evaluate.py
```

## Precomputing Base Logits

Not applicable in this public API-first release.

- There is no shipped local model checkpoint + offline logits pipeline.
- If you need base-logit caching for your private setup, implement it in your internal fork and keep this interface unchanged.

## Architecture & Code Structure

### Tool list (runtime)
Defined in `src/tools/tool_definitions.py`:

- `chat_response`
- `ask_user_confirmation`
- `clarify_experiment_scope`
- `retrieve_knowledge`
- `generate_scientific_draft`
- `reflect_on_protocol`
- `modify_protocol`
- `align_draft_to_automation`
- `generate_machine_code`
- `validate_machine_code`
- `fix_machine_code`
- `add_memory`

### Core files
- `main.py`: interactive CLI entrypoint
- `main_evaluate.py`: core agent runtime (`ProAgent`, session loop, tool execution)
- `src/core/planner_transfer_matrix.py`: FSM state inference and tool plan generation
- `src/core/llms.py`: LLM client + long-memory backend initialization
- `src/prompts/`: planner/tool prompt templates
- `src/tools/`: tool interfaces and tool description payloads

### Capability modules (public shells)
- `src/capabilities/retrieval/knowledge_sources.py`
- `src/capabilities/registry/`
- `src/capabilities/automation/`
- `src/capabilities/verification/engine.py`

These are deliberately simplified extension shells. Replace with your private implementations for production-grade deployment.

### Customization entry points
If you want to adapt for your own lab stack:

1. Retrieval sources and connectors:
   - `src/capabilities/retrieval/knowledge_sources.py`
2. Device/consumable registry and ID mapping:
   - `data/registry/*.json`
   - `src/capabilities/registry/`
3. Draft-to-machine-code mapping policy:
   - `src/capabilities/automation/`
4. Physical rule engine details (`K_p`):
   - `src/capabilities/verification/engine.py`
5. Prompt policies:
   - `src/prompts/prompts.py`
   - `src/prompts/planner.py`


---

## 🔗 The BioProSuite Series

This work is the execution engine of the broader **BioProSuite** initiative (formerly BioPro Project). You can explore our overarching vision and other related research on our project homepage:
🌐 [BioProSuite Homepage](https://yuyangsunshine.github.io/BioPro-Project/)

### Related Work: BioProBench

[cite_start]BioProAgent builds upon and is evaluated using an extended version of **BioProBench**. BioProBench is the first comprehensive dataset and benchmark specifically designed for biological protocol understanding and procedural reasoning.

* 💻 [BioProBench GitHub Repository](https://github.com/YuyangSunshine/bioprotocolbench)
* 🤗 [BioProBench on HuggingFace](https://huggingface.co/BioProBench)

## 📝 Citation

If you find our work or the BioProSuite series helpful for your research, please consider citing our papers:

```bibtex
@article{liu2026bioproagent,
  title={BioProAgent: Neuro-Symbolic Grounding for Constrained Scientific Planning},
  author={Liu, Yuyang and Wang, Jingya and Lv, Liuzhenghao and Tian, Yonghong},
  journal={ACL},
  year={2026}
}

@article{liu2025bioprobench,
  title={BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning},
  author={Liu, Yuyang and Lv, Liuzhenghao and Zhang, Xiancheng and Yuan, Li and Tian, Yonghong},
  journal={arXiv preprint arXiv:2505.07889},
  year={2025}
}

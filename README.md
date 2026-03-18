# BioProAgent: Neuro-Symbolic Grounding for Constrained Scientific Planning

[![Webpage](https://img.shields.io/badge/Project-Webpage-blue.svg)](https://yuyangsunshine.github.io/BioPro-Project/) 
[![Paper](https://img.shields.io/badge/Paper-arXiv:2603.00876-red.svg)](https://arxiv.org/abs/2603.00876)
[![Platform](https://img.shields.io/badge/Demo-AI4S_LAB-success.svg)](https://ai4slab.pkusz.edu.cn/)

> **🎉 News:** > * **[2026-03-18]** 🚀 **Live Demo:** BioProAgent is now officially deployed on the [AI4S LAB platform](https://ai4slab.pkusz.edu.cn/). You can experience the agent and order automated wet-lab experiments directly!
> * **[2026-03-01]** 📄 **Preprint:** Our paper is available on [arXiv](https://arxiv.org/html/2603.00876v1).
> * **[2026-02]** 🏆 **Accepted:** This work has been accepted by the [ICLR 2026 LLA Workshop](https://lifelongagent.github.io/)!
> 
> **⚠️ Note:** The concrete implementation code will be released very soon. Stay tuned!

## 📖 Overview

Large language models (LLMs) have demonstrated significant reasoning capabilities in scientific discovery but struggle to bridge the gap to physical execution in wet-labs[cite: 44]. In these irreversible physical environments, probabilistic hallucinations are not merely incorrect, but can also cause catastrophic equipment damage or experimental failure[cite: 45].

To address this critical execution gap, we propose **BioProAgent**, a training-free neuro-symbolic framework that anchors probabilistic planning in a deterministic Finite State Machine (FSM)[cite: 46]. This controller acts as a safety boundary, enforcing a rigorous "Design-Verify-Rectify" workflow to ensure reliable autonomy[cite: 47].

## ✨ Key Contributions

* [cite_start]**State-Augmented Planning:** BioProAgent uses a deterministic FSM to enforce a strict Design-Verify-Rectify loop[cite: 47]. [cite_start]This ensures that all hardware instructions undergo hierarchical verification for both scientific logic and physical safety before execution[cite: 162].
* [cite_start]**Semantic Symbol Grounding:** To tackle the context bottleneck inherent in complex device schemas, we decouple high-dimensional payloads into symbolic pointers[cite: 48]. [cite_start]This reduces token consumption by ~6× while maintaining 100% resource consistency[cite: 48, 164].
* [cite_start]**Trustworthy Autonomy & Self-Correction:** Evaluated on the extended BioProBench, BioProAgent achieves **95.6% physical compliance** (compared to 21.0% for ReAct) [cite: 49] [cite_start]and an **88.7% success rate in error recovery** (compared to 0% for standard baselines)[cite: 165].

## 🔗 The BioProSuite Series

This work is the execution engine of the broader **BioProSuite** initiative (formerly BioPro Project). You can explore our overarching vision and other related research on our project homepage:
🌐 [BioProSuite Homepage](https://yuyangsunshine.github.io/BioPro-Project/)

### Related Work: BioProBench

[cite_start]BioProAgent builds upon and is evaluated using an extended version of **BioProBench**[cite: 49, 165]. BioProBench is the first comprehensive dataset and benchmark specifically designed for biological protocol understanding and procedural reasoning.

* 💻 [BioProBench GitHub Repository](https://github.com/YuyangSunshine/bioprotocolbench)
* 🤗 [BioProBench on HuggingFace](https://huggingface.co/BioProBench)

## 📝 Citation

If you find our work or the BioProSuite series helpful for your research, please consider citing our papers:

```bibtex
@article{liu2026bioproagent,
  title={BioProAgent: Neuro-Symbolic Grounding for Constrained Scientific Planning},
  author={Liu, Yuyang and Wang, Jingya and Lv, Liuzhenghao and Tian, Yonghong},
  journal={arXiv preprint arXiv:2603.00876},
  year={2026}
}

@article{liu2025bioprobench,
  title={BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning},
  author={Liu, Yuyang and Lv, Liuzhenghao and Zhang, Xiancheng and Yuan, Li and Tian, Yonghong},
  journal={arXiv preprint arXiv:2505.07889},
  year={2025}
}

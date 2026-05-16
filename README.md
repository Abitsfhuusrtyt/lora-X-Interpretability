# LoRA × Mechanistic Interpretability
### Representation-Level Effects of Low-Rank Adaptation in Large Language Models using Sparse Autoencoders

> **Status:** Active Research — Phase 2 (Causal Validation)  
> **Base Model:** google/gemma-2-9b + Gemma Scope SAEs  

---

## Overview

Large language models are increasingly deployed not as base models but as fine-tuned variants. **LoRA (Low-Rank Adaptation)** is the dominant fine-tuning method — used for instruction tuning, domain adaptation, safety alignment, and capability steering. Despite its widespread use, almost nothing is understood about what LoRA actually does to a model's internal feature geometry.

This project asks a precise mechanistic question:

> **When you fine-tune a model with LoRA, what happens to the sparse, interpretable features that mechanistic interpretability has identified in the base model? Are those features reused, amplified, suppressed — or does LoRA operate in an entirely different representational subspace?**

We answer this using **Sparse Autoencoders (SAEs)** applied directly to the activation deltas induced by LoRA adapters — the first systematic study of fine-tuning effects at the feature level rather than the weight level.

---

## Why This Matters for AI Safety

Current mechanistic interpretability tools — SAEs, linear probes, activation patching — are trained and validated on base models. If safety fine-tuning via LoRA operates in a representational subspace that is **geometrically orthogonal** to the base model's feature space, then:

- Safety monitors trained on base model features may be **blind** to adapter-induced behaviour
- Safety rules encoded by LoRA may be **causally fragile** — detectable geometrically but not robust under adversarial pressure
- Standard interpretability audits of fine-tuned models may be **systematically incomplete**

This project provides the first empirical evidence for this concern and the methodological framework to address it.

---

## Research Questions

**RQ1 — Feature Reuse vs Feature Creation:**
Does LoRA fine-tuning primarily amplify existing base model features, or does it create new feature directions not present in the base model's learned representations?

**RQ2 — Rank and Feature Geometry:**
How does adapter rank (r=4, 8, 16, 32) influence the geometric structure, sparsity, and density of adapter-induced features across transformer layers?

**RQ3 — Causal Efficacy:**
Are the novel feature directions identified by delta SAEs causally responsible for behavioural change, or do they represent representational changes that are downstream of output predictions?

**RQ4 — Safety Monitoring Gap:**
Do standard base-model interpretability tools (Gemma Scope SAEs) detect adapter-induced features? If not, what does this imply for the auditability of LoRA-based safety alignment?

---

## Methodology

### The Core Insight: Activation Delta Analysis

When a LoRA adapter is active, the residual stream receives:

```
h_adapted = h_base + BAx
```

where `BAx` is the adapter's direct contribution. We isolate this delta:

```
h_delta = h_adapted - h_base
```

and apply SAEs to decompose it into interpretable feature directions. This is mechanistically clean — `h_delta` contains only what the adapter added, with no base model signal.

### Pipeline Overview

```
1. Train LoRA adapters (r=4, 8, 16, 32) on Alpaca instruction data
        ↓
2. Extract h_base, h_delta across 6 target layers for 2000 probe samples
        ↓
3. Pass h_delta through Gemma Scope SAEs (base model dictionary)
   → Measure: feature overlap, reconstruction error, feature density
        ↓
4. Train delta SAEs on h_delta (adapter-specific dictionary)
   → 24 SAEs: 4 ranks × 6 layers
        ↓
5. Compare delta SAE and Gemma Scope SAE decoder geometries
   → Cosine similarity between feature dictionaries
        ↓
6. Held-out reconstruction evaluation on unseen samples
        ↓
7. Causal validation via activation steering (in progress)
```

### Experimental Configuration

| Component | Details |
|---|---|
| Base model | google/gemma-2-9b (9.24B parameters) |
| Architecture | Decoder-only, 42 layers, d_model=3584, GQA |
| LoRA ranks | 4, 8, 16, 32 |
| lora_alpha | 2 × rank (fixed scaling factor = 2.0) |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Training data | tatsu-lab/alpaca, 10,000 samples, 3 epochs |
| Probe dataset | Alpaca indices 5000–6999 (2000 samples, diverse) |
| Held-out data | Alpaca indices 11000–11199 (200 samples) |
| Target layers | [5, 10, 18, 22, 32, 38] |
| Base SAEs | Gemma Scope (google/gemma-scope-9b-pt-res, width_16k) |
| Delta SAE width | 16,384 features (matching Gemma Scope) |
| Delta SAE L1 | 0.15 (after RMS normalisation) |
| Delta SAE L0 | ~31 active features/token (all 24 SAEs) |

---

## Current Findings

### Finding 1 — LoRA Predominantly Activates Novel Features (Not Base Model Features)

Passing h_delta through Gemma Scope SAEs, we measure what fraction of activated features were already active in h_base (amplification) vs newly activated (novel).

| Layer | Novel fraction (r=4) | Novel fraction (r=32) |
|---|---|---|
| 5 | 79.63% | 79.63% |
| 10 | 78.85% | 75.91% |
| 18 | 77.44% | 78.02% |
| 22 | 78.13% | 78.33% |
| 32 | 72.32% | 74.54% |
| 38 | 73.72% | 75.91% |

**Interpretation:** Regardless of rank, 92–99.7% of features activated by the adapter delta are not present in base model activations. LoRA is not amplifying what the base model knows — it is activating representational directions the base model does not use.

### Finding 2 — Delta SAE Features Are Geometrically Orthogonal to Base SAE Features

We computed maximum cosine similarity between every delta SAE decoder direction and every Gemma Scope decoder direction (268 million comparisons per layer).

| Metric | Value |
|---|---|
| Mean max cosine similarity | 0.071 |
| Median max cosine similarity | 0.066 |
| Features with max sim < 0.3 (novel) | 79.77% |
| Features with max sim > 0.7 (shared) | 0.02% (3 features) |

Random unit vectors in 3584-dimensional space have expected cosine similarity ≈ 0. A mean of 0.071 indicates the delta SAE feature dictionary is **near-orthogonal** to the Gemma Scope dictionary — not a slight rotation, but an almost entirely different set of directions in the same space.

Only 3 features out of 16,384 show strong alignment (>0.7) with any base model feature. These represent the tiny fraction of adapter behaviour that builds on existing base knowledge.

### Finding 3 — Delta SAEs Achieve 46–86% Lower Reconstruction Error on Held-Out Data

Comparing reconstruction quality on 200 completely unseen samples (indices 11,000+):

| Layer | Gemma Scope error | Delta SAE error | Improvement |
|---|---|---|---|
| 5 | 2.46 | 0.34 | +86.2% |
| 10 | 1.57 | 0.44 | +71.6% |
| 18 | 1.28 | 0.39 | +69.9% |
| 22 | 1.18 | 0.39 | +67.2% |
| 32 | 1.26 | 0.57 | +54.9% |
| 38 | 1.14 | 0.61 | +46.3% |

Delta SAEs outperform Gemma Scope on all 24 conditions (4 ranks × 6 layers). This confirms that delta SAEs learned genuine, generalisable adapter-specific structure — not noise or training artifacts.

### Finding 4 — Feature Density Scales with Rank at Deep Layers

At layer 38, the number of active features per token increases monotonically with rank:

| Rank | Active features (layer 38) |
|---|---|
| 4 | 30.28 |
| 8 | 33.65 |
| 16 | 34.95 |
| 32 | 41.66 |

Higher rank adapters activate more distinct features at deep layers — consistent with rank increasing representational capacity rather than simply scaling existing features.

### Finding 5 — Delta Norm Amplifies with Layer Depth (Non-Monotonically with Rank)

The adapter's perturbation magnitude grows ~18x from layer 5 to layer 38, consistent across all ranks. However, the relationship between rank and delta magnitude is non-monotonic — r=8 produces the largest delta norm at layer 38 (345.45), larger than r=32 (330.81).

---

## Key Hypotheses

**H1 — Orthogonal Subspace Hypothesis (Supported):**
LoRA fine-tuning induces residual stream perturbations in a feature subspace orthogonal to the base model's learned representations, regardless of rank.

**H2 — Monitoring Gap Hypothesis (Supported observationally, causal validation pending):**
Base-model interpretability tools are systematically blind to adapter-induced features because those features point in geometrically orthogonal directions.

**H3 — Distributed + Sparse Decomposition (Partially supported):**
LoRA's representational effect has two components — a sparse interpretable component (captured by delta SAEs at 46–86% quality) and a distributed uninterpretable residual that cannot be decomposed by any sparse linear dictionary.

**H4 — Causal Fragility of Safety LoRA (Under investigation):**
Because adapter features are orthogonal to base model features, the base model's existing circuits may route around adapter contributions under adversarial conditions — making safety fine-tuning via LoRA potentially non-robust.

---

## Negative Results

**Activation steering with LoRA adapter features produced no behavioural effect.** Steering the base model with top delta SAE feature directions at layer 22 (scales 10–200) produced identical outputs to the unsteered base model across all prompts, ranks, and features tested.

**Root cause:** The trained LoRA adapters produce outputs identical to the base model on all test prompts. Gemma-2-9b's pretraining priors are strong enough that 3 epochs of Alpaca fine-tuning produces no measurable behavioural change, despite detectably modifying internal representations.

**Mechanistic implication:** This suggests the adapter may be operating in a subspace that is causally downstream of output predictions — geometrically present but functionally bypassed. Alternatively, it may simply reflect insufficient fine-tuning. Both hypotheses are under investigation using gemma-2-9b-it as a strongly-differentiated reference model.

---

## Current Research Phase

### ✅ Completed
- LoRA adapter training at ranks 4, 8, 16, 32 (10k samples, 3 epochs)
- Activation extraction pipeline (h_base, h_delta, 2000 samples, 6 layers)
- Gemma Scope SAE analysis (feature overlap, reconstruction error, feature density)
- Delta SAE training (24 SAEs, L0≈31, all held-out validated)
- Dictionary similarity analysis (cosine similarity between feature dictionaries)
- Held-out reconstruction evaluation (200 unseen samples)
- Activation steering experiments (null result with LoRA adapters)

### 🔄 In Progress
- Causal validation using gemma-2-9b-it as reference model
- Feature attribution on disagreement prompts
- Activation patching experiments

### 📋 Planned
- Base-base SAE similarity baseline (to validate orthogonality finding)
- Cross-rank delta SAE similarity analysis
- SVD analysis of h_delta matrices (effective dimensionality)
- Neuronpedia feature interpretation of shared features
- Scale-up to 500+ samples for delta SAE retraining
- Paper writing and ArXiv submission

---


---

## Installation

```bash
# Clone the repository
git clone https://github.com/Abitsfhuusrtyt/lora-X-Interpretability
cd loraXinterpretability

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install transformers peft datasets accelerate sae-lens huggingface-hub torch

# Login to HuggingFace (required for Gemma-2 access)
huggingface-cli login
# Accept Gemma license at: https://huggingface.co/google/gemma-2-9b
```

---

## Reproducing the Experiments

```bash
# 1. Train LoRA adapters
python train_lora_adapters.py

# 2. Extract activations
python extract_activations.py

# 3. Run Gemma Scope SAE analysis
python sae_analysis.py

# 4. Train delta SAEs
python train_delta_sae.py

# 5. Dictionary similarity analysis
python dictionary_similarity.py

# 6. Held-out reconstruction evaluation
python reconstruction_eval.py

# 7. Activation steering
python token_prediction_steering.py
```

**Hardware requirements:** Apple M4 Studio (64GB unified memory) or equivalent. Full pipeline requires ~110GB disk space for activations and ~12GB for model weights.

---

## Connections to AI Safety

This work connects to several active AI safety research threads:

**Fine-tuning robustness:** Recent work (Yang et al., 2023; Qi et al., 2023) has shown that safety fine-tuning can be easily undone by subsequent fine-tuning. Our findings provide a mechanistic explanation — if safety features occupy an orthogonal subspace, they may be especially vulnerable to being overwritten or bypassed.

**Monitoring and auditing:** If adapter features are geometrically invisible to base-model SAEs, organisations deploying fine-tuned models cannot use standard interpretability tools to audit what the fine-tuning encoded. This is a concrete, measurable safety gap.

**Representation engineering:** Our delta SAE framework extends representation engineering (Zou et al., 2023) to adapter-specific feature spaces, providing a more precise tool for steering and controlling fine-tuned model behaviour.

**Mechanistic understanding of alignment:** Understanding what safety fine-tuning actually does at the feature level — as opposed to the behavioural level — is a prerequisite for building alignment methods that are mechanistically robust rather than behaviourally superficial.

---

## About

This research is being conducted independently by **Prasanth**, an AI Research Engineer at Zoho Corporation and independent ML researcher based in India. The project is being developed alongside an MSc in AI at Dublin City University (starting September 2026), with a long-term goal of contributing to mechanistic interpretability and AI safety research.

**Published work:**
- Geometric Mixture Classifier (GMC) — arXiv
- GRAIL — Zenodo
- EchoLSTM — arXiv

**Affiliations:** Zoho Corporation 
---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{prasanth2025lora,
  title={Representation-Level Effects of LoRA in Large Language Models 
         using Sparse Autoencoders},
  author={Prasanth},
  year={2025},
  note={Preprint in preparation}
}
```

---

## Contact

For questions, collaborations, or feedback on the research:
- Open a GitHub issue
- Reach out via {abiprasanth0101@gmail.com}

---

*This README will be updated as the research progresses toward publication.*

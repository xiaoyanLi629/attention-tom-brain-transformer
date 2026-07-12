# Attention Mechanisms for Theory of Mind

**Weak Brain-Transformer Alignment Despite Behavioral Success**

IEEE BIBM 2026 submission (revised from CogSci 2026 #3240, rejected).

---

## Summary

Theory of Mind (ToM) — attributing mental states to others — is a critical test of
whether language models develop genuine social understanding. We compare ToM
processing in human brains (N=58, fMRI) against four transformer models
(355M–7B parameters) using a modality-matched paradigm: participants listen to
**social vs. physical** descriptions of the same animated shapes, while the models
process the identical text transcripts.

**The headline is a dissociation.** Transformers pass the behavioral tests but do
*not* align with human neural representations:

| Finding | Result |
|---|---|
| Behavioral ToM (false belief, faux pas, intention) | 7B models **75–83%**; GPT-2 variants near chance (~50%) |
| Linear probing (social vs. physical from hidden states) | **81–88%** accuracy |
| Causal attention-head ablation | Top-5% of heads (mid-layers) drive mental-state prediction; a **single GPT-2-Medium head carries 25%** of the baseline logit gap |
| **Time-resolved encoding (brain ↔ model)** | **Weak: best r ≈ 0.04** (noise ceiling ≈ 0.16–0.25) |
| **Social-vs-physical encoding advantage, after BH-FDR** | **Disappears entirely — 0/786 contrasts survive** |
| Brain itself (voxel-wise) | Clear fronto-temporal social > physical pattern |
| Attention sparsity | Slightly higher for social narratives (Gini 0.66–0.77) |

> **Note on an earlier version of this README.** It reported strong RSA alignment
> (r = 0.42–0.62) and a large causal circuit (198–425 heads). Those numbers came
> from the rejected CogSci pipeline and **do not survive** the corrected analysis.
> The current, FDR-corrected result is the *weak-alignment null* above. If you are
> reviewing this repository against the manuscript, the manuscript is correct.

## Data

**OpenNeuro ds002345** (Narratives — "Shapes" social-attribution task).
59 subjects; sub-115 excluded → **N=58 analyzed**.

- `data/ds002345/fmri/` — preprocessed BOLD
- `data/ds002345/transcripts/` — `shapessocial_transcript.txt`, `shapesphysical_transcript.txt`
  (the text the models see; this is what makes the paradigm modality-matched)
- `data/ds000109/` — a second ToM dataset, kept as a backup. **Not used** by any
  script in the current pipeline or by the manuscript.

> The earlier README claimed HCP data with N=20. That was the CogSci-era design and
> is no longer accurate.

## Models

Four transformers, all in `models/huggingface_cache/`:

| Model | Params |
|---|---|
| GPT-2-Medium | 355M |
| GPT-2-XL | 1.5B |
| Mistral-7B | 7B |
| Qwen2-7B | 7B |

These are the four reported in the manuscript (see `MODEL_ORDER` in
`scripts/s07_visualization.py`). `configs/config.py` additionally lists
DeepSeek-MoE-16B and Phi-3-Mini; **DeepSeek's weights were never downloaded**
(only config/tokenizer stubs are on disk) and neither model appears in the paper.

## Pipeline

```bash
python run_pipeline.py                # all stages
python run_pipeline.py --stage 3      # a single stage
python run_pipeline.py --visualize    # stage 7 only
```

| Stage | Script | What it does |
|---|---|---|
| 1 | `s01_fmri_preprocessing.py` | fMRI preprocessing, ROI time-series extraction |
| 1b | `s01b_wholebrain_contrast.py` | Whole-brain social > physical group t-map |
| 2 | `s02_llm_extraction.py` | Time-resolved LLM feature extraction from transcripts |
| 3 | `s03_encoding_models.py` | **Encoding models — the core brain↔model alignment test** |
| 4 | `s04_attention_analysis.py` | Attention sparsity / specialization |
| 5 | `s05_causal_analysis.py` | Causal attention-head ablation |
| 6 | `s06_probing.py` | Linear probing: social vs. physical from hidden states |
| 7 | `s07_visualization.py` | Publication figures (PDF) |
| 8 | `s08_behavioral_tom.py` | Behavioral ToM benchmark (false belief, faux pas, intention) |

Helpers, not pipeline stages: `s01_parallel_rerun.py` (parallel MNI resampling),
`patch_stats.py`.

## Layout

```
Attention_ToM/
├── run_pipeline.py
├── configs/config.py           # paths, model list, ROIs
├── scripts/                    # s01–s08 (see table above)
├── data/                       # ds002345 (used), ds000109 (backup, unused)
├── models/huggingface_cache/   # 4 transformers (+2 unused entries in config)
├── results/
├── logs/
├── IEEE_manuscript/            # ← current BIBM submission
├── manuscript/                 # CogSci version (rejected, superseded; gitignored)
├── REVISION_PLAN.md            # internal notes on the CogSci → BIBM rewrite
└── sources/                    # private research notes (gitignored)
```

## Citation

```bibtex
@inproceedings{attention_tom_2026,
  title     = {Attention Mechanisms for Theory of Mind: Weak Brain-Transformer
               Alignment Despite Behavioral Success},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

---

*Last updated: 2026-07 · N=58 · ds002345 · 4 transformers · 8 pipeline stages*

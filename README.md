# Attention Mechanisms for Theory of Mind

**Weak Brain-Transformer Alignment Despite Behavioral Success**

IEEE BIBM 2026, regular paper B323 (camera-ready). Camera-ready source and PDF:
`IEEE_manuscript/camera_ready/`.

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
| Linear probing (social vs. physical from hidden states) | **89–98%** accuracy (bag-of-words baseline: 74%) |
| **Time-resolved encoding (brain ↔ model)** | **Weak: best r = 0.047** (mPFC, Qwen2-7B; ≈19% of the noise-ceiling upper bound) |
| Best-layer encoding vs. zero (leave-one-subject-out layer choice, BH-FDR) | **0/48 cells survive** (smallest q = 0.09) |
| Robustness | Unchanged across HRF lags 0–6 TRs, kernel ridge / MLP encoders, 8 mm and 6 mm ROIs, GPT-2 truncation |
| Social-vs-physical encoding contrast | 86% of 792 contrasts favour social, **0 survive BH-FDR** |
| Brain itself (voxel-wise) | Clear fronto-temporal social > physical pattern |
| Attention sparsity | Reliably but slightly sparser for social narratives (paired over layers) |
| Causal attention-head ablation | Individual heads with large effects (GPT-2-Medium L6H1); the top-5% set is only partly stable across 19 prompts |

> **Camera-ready correction (2026-09).** Stage 2 originally mis-aligned model
> features and fMRI: word-initial tokens were not matched to words (byte-level /
> SentencePiece offsets include the leading space, so only ~15% of words were
> mapped), the story-start search silently fell back to offset 0 (every TR got
> features from ~80 words earlier), and the 4.5 s audio onset in the scan was
> ignored. All three are fixed (`scripts/s01_fmri_preprocessing.py`,
> `scripts/s02_llm_extraction.py`); the story now spans exactly the 272 / 270 TRs
> of the dataset metadata. PCA uses an exact SVD so results are reproducible.
> Numbers above come from the corrected run; the reviewer-requested analyses are in
> `scripts/camera_ready/` (`run_cpu_chain.sh` reruns the CPU part).

## Data

**OpenNeuro ds002345** (Narratives — "Shapes" social-attribution task).
59 subjects; sub-115 excluded → **N=58 analyzed**.

- `data/ds002345/fmri/` — preprocessed BOLD
- `data/ds002345/transcripts/` — Whisper output; `*_words.txt` (word-level timestamps)
  is the text the models see; this is what makes the paradigm modality-matched
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
│   └── camera_ready/           # reviewer-requested analyses (cr00–cr11, run_cpu_chain.sh)
├── data/                       # ds002345 (used), ds000109 (backup, unused)
├── models/huggingface_cache/   # 4 transformers (+2 unused entries in config)
├── results/
├── logs/
├── IEEE_manuscript/            # submitted version; camera_ready/ = final PDF + source
├── manuscript/                 # CogSci version (rejected, superseded; gitignored)
├── REVISION_PLAN.md            # internal notes on the CogSci → BIBM rewrite
└── sources/                    # private research notes (gitignored)
```

## Citation

```bibtex
@inproceedings{li2026attention_tom,
  title     = {Attention Mechanisms for Theory of Mind: Weak Brain-Transformer
               Alignment Despite Behavioral Success},
  author    = {Li, Xiaoyan and Jiang, Cuicui and Chen, Jiaoping and Yang, Rumei
               and Liu, Xingyue and Wei, Jiaxuan},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

---

*Last updated: 2026-09 (camera-ready) · N=58 · ds002345 · 4 transformers · 8 pipeline stages*

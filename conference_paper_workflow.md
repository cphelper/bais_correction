## Conference-ready workflow (for your SWH/SLA bias-correction project)

This workflow turns your existing analysis pipeline (SWH + SLA satellite-guided correction) into a **conference-level paper + reproducibility package**. It is written to match your current repo artifacts (e.g., `main_ijcnn.tex`, `workflow.md`, `sla_analysis_report.md`, analysis scripts, and `plots/` + `sla_plots/` outputs).

---

## A. End-to-end workflow (big picture)

```mermaid
flowchart TD
  A[Define venue + constraints\nIEEE WCCI/IJCNN vs ICML/ICLR/NeurIPS-style] --> B[Lock problem statement + claims\nWhat improves, why it matters]
  B --> C[Dataset + preprocessing spec\nSWH: cowcliphs.nc + io-altimeter.nc\nSLA: altimeter_sla/ + CMIP6 zos]
  C --> D[Leakage-proof evaluation design\nTime split + spatial split + extremes]
  D --> E[Baselines + ablations\nStatistical + ML baselines\nFeature/model ablations]
  E --> F[Train models\nRF (existing) + baselines\nCalibrate hyperparameters]
  F --> G[Evaluate + uncertainty\nRMSE/Bias/R²/Corr\nTail metrics + CIs/bootstraps]
  G --> H[Figures that tell the story\nMaps, time series, scatter, tables]
  H --> I[Write paper draft\nMain claims supported in main body]
  I --> J[Reproducibility package\nOne-command rerun + env + scripts]
  J --> K[Compliance pass\nAnonymization + page limits + checklists]
  K --> L[Submission + rebuttal plan\nPrepare responses + additional analyses]
```

---

## B. Phase 0 — Choose the target “ruleset” (do this first)

Different venues impose different **page limits, anonymization rules, and required artifacts**. Pick one target and make everything fit.

- **IEEE WCCI / IJCNN-style (matches `main_ijcnn.tex`)**
  - IEEE template, usually **6 pages** (often **up to 8** with extra-page fees).
  - **Double-blind** (remove authors/affiliations; avoid identity leaks).
  - Add an **Open Science note** in the last paragraph of the introduction (promise to share code/data upon acceptance, or justify why not).
  - If AI-generated text was used, **disclose in Acknowledgements** with a citation to the AI tool.

- **ICML-style**
  - Strict **8-page main body**; refs + appendices can be longer (single PDF).
  - LaTeX-only; **strict anonymization**; if it leaks identity, can be rejected.
  - Any material *critical to evaluation* must be in the **main body**, not only in appendix/supplement.

- **ICLR-style**
  - **≤9 pages main text** at submission (refs excluded).
  - Strongly recommended: **Reproducibility Statement** + **Ethics Statement** (not counted toward page limit).
  - If LLMs materially contributed to ideation/writing, include an **LLM usage section** (usually in appendix).

- **NeurIPS-style (use as a quality gate even if not submitting there)**
  - The **NeurIPS Paper Checklist** is mandatory for NeurIPS submissions (desk reject if missing).
  - Strong expectations: limitations, reproducibility, compute reporting, licensing notes.

---

## C. Phase 1 — Lock the story: problem, claims, and contributions (1–2 days)

### C1) Define the problem precisely

- **SWH**: bias in CMIP6-proxy SWH vs satellite altimeter SWH over the Indian Ocean; goal is bias-corrected projections for future scenarios.
- **SLA**: bias in CMIP6 `zos`-derived anomalies vs satellite SLA in ROI; goal is corrected anomaly time series/maps for downstream use.

### C2) Write a “claim sheet” (should fit in 6–8 bullets)

Example claim types that reviewers accept:
- **Raw model bias exists and is structured** (spatial + seasonal).
- **Method learns state-dependent corrections** using `model_value`, `lat`, `lon`, `month`.
- **Generalizes to unseen years** (your existing temporal split is a strong point).
- **Improves extremes** (tail behavior) *or* explicitly states limitations on extremes.
- **Reproducible**: scripts + environment + instructions recreate key figures/tables.

### C3) Define your contributions

Keep these concrete and testable:
- A leakage-resistant evaluation design (time + space).
- A baseline suite (statistical + ML) and ablations.
- A reproducible pipeline and figures.

---

## D. Phase 2 — Data, preprocessing, and comparability (2–5 days)

### D1) Dataset documentation checklist (put in the paper)

For each dataset (SWH model, SWH obs, SLA model, SLA obs):
- **File(s)** and variable names
- **Spatial grid** (resolution, coordinate conventions)
- **Time coverage** and resampling method (daily → monthly, etc.)
- **Units** and anomaly definition (important for SLA)
- **Masking** and missing-data treatment

### D2) “Comparability contract” (avoid reviewer nitpicks)

Document these decisions explicitly:
- Regridding method (bilinear vs nearest-neighbor; curvilinear handling for `zos`)
- Temporal alignment (monthly means, time stamps, calendars)
- What region is evaluated (ROI definition for SLA)
- How anomalies are defined (e.g., subtract mean of training period)

---

## E. Phase 3 — Evaluation design (this is the biggest difference between “project” and “conference paper”) (3–7 days)

### E1) Mandatory split: time-based generalization

Use your current split as the default:
- **Train**: 2015–2018
- **Test**: 2019–2020

Explain why: simulates future generalization, reduces leakage vs random split.

### E2) Add at least one spatial generalization test

Pick one:
- **Blocked tiles**: hold out spatial tiles (e.g., 10°×10° or 5°×5° bins).
- **Basin holdout**: hold out one region (Arabian Sea, Bay of Bengal, Southern Ocean, Equatorial Indian).
- **Lat-band holdout**: hold out lat ranges to test extrapolation.

### E3) Tail/extremes evaluation (especially important for waves)

Report at least one:
- Error at **95th/99th percentile** of SWH
- Conditional RMSE for high-SWH regime (e.g., SWH > 3 m)
- Quantile/QQ comparison focused on upper tail

### E4) Report variation / uncertainty

Options:
- **Bootstrap** over time blocks or spatial blocks → confidence intervals for metrics
- Multiple seeds for stochastic models (RF can be seeded; other baselines may vary)

---

## F. Phase 4 — Baselines + ablations (3–7 days)

### F1) Baselines (minimum set that passes tough review)

Include at least:
- **No correction** (raw CMIP6)
- **Mean bias correction** (global and/or seasonal)
- **Linear regression** (same features as RF)
- **Quantile mapping / QDM** (grid-wise or seasonal)
- **Modern tabular ML** (e.g., Gradient Boosting)

### F2) Ablations (to justify your design)

Feature ablations (examples):
- RF with only `model_value`
- RF with `model_value + month`
- RF with `model_value + lat + lon`
- RF with all features

Model ablations (examples):
- Tree depth, number of trees
- Different loss/objective if using boosting

---

## G. Phase 5 — Figures that win (2–5 days)

Create figures so a reviewer can understand the paper by reading figures + captions.

### Recommended figure set (high ROI)

- **Fig 1**: Pipeline schematic (can adapt `workflow.md`)
- **Fig 2**: Bias/RMSE maps (raw vs corrected; same color scale)
- **Fig 3**: Regional time series (obs vs raw vs corrected)
- **Fig 4**: Scatter + 1:1 line (and show tail region explicitly)
- **Fig 5**: Baseline comparison table (metrics + CIs if possible)
- **Fig 6**: Interpretability (feature importance; optional partial dependence)

Rules:
- Use consistent colormaps and units.
- Keep readable at 1-column width (IEEE format).
- Captions must state **what improves** and **by how much**.

---

## H. Phase 6 — Write the paper (5–10 days, overlaps with Phase 3–5)

### Suggested section order (fits your `main_ijcnn.tex` structure)

- **Introduction**: problem + stakes + contributions + open science note (IEEE).
- **Related Work**: bias correction (QM/QDM) + ML correction + spatiotemporal generalization.
- **Data and Methods**:
  - datasets + preprocessing
  - features/targets
  - models (RF + baselines)
  - evaluation protocol (time + space + tails)
- **Results**:
  - raw uncertainty diagnostics
  - correction performance (overall + seasonal + regional)
  - baseline/ablation summary
- **Discussion**: why it works, where it fails, how to use it for future scenarios.
- **Limitations** (explicit): distribution shift, extremes, missing covariates, curvilinear interpolation artifacts.
- **Conclusion**: what changed, why it matters, next steps (hybrid QM+ML, physics-informed constraints, longer altimeter record).

Critical rule (ICML/ICLR/NeurIPS norms):
- If a result is central to your claim, ensure it’s **in the main body**, not only in appendix.

---

## I. Phase 7 — Reproducibility package (1–3 days)

### Minimum “re-run” contract

Provide:
- Exact environment spec (`requirements.txt` or `environment.yml`)
- Exact commands to regenerate:
  - Key metric table(s)
  - 4–6 main figures
- Deterministic seeding strategy
- Notes on compute (CPU/GPU, RAM, runtime)

### Recommended artifact layout

```text
repro/
  README.md              # exact commands + expected outputs
  environment.yml        # or requirements.txt
  scripts/
    run_all.sh           # optional: one command pipeline
  outputs/
    figures/             # generated plots
    tables/              # generated metrics
```

---

## J. Phase 8 — Anonymization + compliance (desk-reject prevention) (0.5–1 day)

### J1) Anonymization checklist (double-blind)

- Remove author names/affiliations/acknowledgements for submission.
- Avoid repository links unless anonymous (and immutable during review if required).
- Avoid “we at ISRO…” wording; use neutral third-person phrasing.
- Scrub figure metadata if it contains usernames/paths.

### J2) Page-limit checklist

- IEEE: compress to 6 pages (or 8 with fees); prioritize core figures.
- ICML: main body ≤8 pages.
- ICLR: main text ≤9 pages.

### J3) Mandatory/expected add-ons (depending on venue)

- IEEE WCCI/IJCNN: Open science note in intro; AI-text disclosure in acknowledgements if used.
- NeurIPS: paper checklist appended after references/appendix (mandatory for NeurIPS).
- ICLR: optional ethics + reproducibility statements (recommended).

---

## K. Phase 9 — Rebuttal readiness (if applicable)

Pre-write answers for common reviewer questions:
- “Is there leakage from using lat/lon?” → justify + show spatial holdout results.
- “Why RF over physics-based correction?” → baseline comparisons + interpretability.
- “How does it handle extremes?” → tail metrics; honest limitations.
- “Reproducibility?” → point to artifact zip + commands.

---

## L. Mapping to your current repo

Use these as anchors:
- **Existing method flow**: `workflow.md` (SWH correction pipeline).
- **SLA report narrative**: `sla_analysis_report.md`.
- **Primary paper drafts**: `main_ijcnn.tex`, `main.tex`.
- **Plots**: `plots/`, `sla_plots/`.
- **Analysis scripts**: `analyze_sla.py`, `analyze_and_report.py`, `create_additional_plots.py`, `create_ml_analysis_plots.py`.


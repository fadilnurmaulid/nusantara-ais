# NUSANTARA-AIS

**Context-aware Maritime Anomaly Detection using AIS and Spatial Intelligence**

A research-ready, end-to-end implementation of a maritime anomaly detection
framework that fuses a physics-informed AIS Reliability Index, a
spatially-aware heterogeneous graph, and a HetGAT + Graph AutoEncoder model
into a continuous **Context-aware Maritime Anomaly Score** in `[0, 1]`.

> The model **does not** classify illegal fishing directly. It produces a
> continuous, ranked anomaly score that supports human triage of suspicious
> maritime activity: illegal fishing, AIS blackout, AIS spoofing, loitering,
> protected-area intrusion, and other abnormal vessel behavior.

Built for GEMASTIK 2026. Datasets referenced throughout: AIS (Global Fishing
Watch), BMKG (wave/wind/weather), bathymetry, coastline, Indonesian ports,
and the **National Shipping Lane Network** (BIG spatial planning data). EEZ
is used as an optional layer.

---

## 1. Why this repository is structured this way

Every stage in the assignment's pipeline diagram is a real, separately
testable Python module — not a monolithic script:

```
Raw AIS
  -> Cleaning                              src/nusantara_ais/data/cleaning.py
  -> Temporal Alignment                    src/nusantara_ais/data/temporal.py
  -> Trajectory Segmentation               src/nusantara_ais/data/trajectory.py
  -> Spatial Feature Engineering           src/nusantara_ais/features/{dynamic,spatial,historical}.py
  -> AIS Reliability Index (ARI)           src/nusantara_ais/features/reliability.py      [Contribution 1]
  -> Heterogeneous Graph Construction      src/nusantara_ais/graph/{nodes,edges,builder}.py [Contribution 2]
  -> HetGAT                                src/nusantara_ais/models/hetgat.py
  -> Graph AutoEncoder                     src/nusantara_ais/models/gae.py
  -> Context-aware Maritime Anomaly Score  src/nusantara_ais/models/anomaly_score.py        [Contribution 3]
  -> Risk Assessment                       src/nusantara_ais/models/anomaly_score.py (risk_tier)
```

`src/nusantara_ais/pipeline.py` is the single reference implementation
tying every stage together end-to-end, and is what `scripts/run_pipeline.py`
calls.

## 2. Directory structure

```
nusantara-ais/
├── configs/                      # (optional) serialized experiment configs (YAML)
├── data/
│   ├── raw/                      # drop real GFW/BMKG/BIG files here (see schemas below)
│   ├── processed/                # intermediate cached artifacts (optional)
│   └── synthetic/                # auto-generated physically-realistic stand-in data
├── src/nusantara_ais/
│   ├── config.py                 # single typed dataclass config (paths, CRS, ARI, graph, model, training)
│   ├── pipeline.py                # end-to-end orchestrator
│   ├── utils/
│   │   └── reproducibility.py     # seeding, device selection, logging
│   ├── data/
│   │   ├── ingestion.py           # unified real/synthetic loader façade
│   │   ├── cleaning.py            # schema enforcement, bounds/duplicate/kinematic filtering
│   │   ├── coordinate.py          # CRS transforms, haversine, bearing
│   │   ├── temporal.py            # fixed-cadence resampling, gap-aware interpolation, blackout flags
│   │   ├── trajectory.py          # segmentation into voyages
│   │   ├── spatial_join.py        # nearest port/lane/coastline, bathymetry, EEZ/MPA containment
│   │   └── synthetic.py           # synthetic AIS + BMKG + bathymetry + coastline + ports + lanes generator
│   ├── features/
│   │   ├── dynamic.py             # speed/heading/acceleration/turning-rate/gap features
│   │   ├── spatial.py             # spatial-join feature orchestration
│   │   ├── reliability.py         # *** AIS Reliability Index (Contribution 1) ***
│   │   ├── historical.py          # revisit/anchoring/loitering/behavioral aggregates
│   │   └── engineer.py            # runs all feature stages in order
│   ├── graph/
│   │   ├── nodes.py                # vessel_state / port / shipping_lane_segment / grid_cell node features
│   │   ├── edges.py                # all 7 edge types + distance-based weighting
│   │   ├── builder.py              # *** heterogeneous graph + snapshot builder (Contribution 2) ***
│   │   └── dataset.py              # PyG Dataset wrapper + chronological train/val/test split
│   ├── models/
│   │   ├── hetgat.py               # Heterogeneous Graph Attention Network encoder
│   │   ├── gae.py                  # Graph AutoEncoder (feature + structure + ARI-consistency losses)
│   │   ├── anomaly_score.py        # *** Context-aware Maritime Anomaly Score (Contribution 3) ***
│   │   ├── full_model.py           # HetGAT + GAE composition
│   │   └── baselines.py            # IsolationForest, OC-SVM, XGBoost, GraphSAGE, GAT
│   ├── training/
│   │   ├── train.py                # training loop: AMP, grad clipping, cosine LR, early stopping, checkpointing
│   │   └── validate.py             # inference + anomaly-score computation over a snapshot dataset
│   ├── inference/
│   │   └── infer.py                # score new raw AIS with a trained checkpoint (train/serve feature parity)
│   └── evaluation/
│       ├── metrics.py              # ROC-AUC/PR-AUC, tier recall/precision, FP/FN analysis, runtime/complexity
│       ├── ablation.py             # baseline comparison, ablation study, feature importance, ARI sensitivity,
│       │                           # embedding analysis, attention analysis
│       └── visualization.py        # every figure in the evaluation suite, saved to outputs/figures/
├── scripts/
│   ├── run_pipeline.py             # CLI: run everything end-to-end, write experiment_report.json
│   └── generate_synthetic_data.py  # CLI: (re)generate synthetic datasets only
├── tests/
│   └── test_core.py                # unit tests: ARI bounds/behavior, distances, graph shapes, cleaning
├── outputs/
│   ├── checkpoints/                # best model checkpoint (.pt) with full config + optimizer state
│   ├── figures/                    # all generated PNG figures
│   ├── logs/                       # per-run log files
│   └── reports/                    # experiment_report.json (all metrics, ready for the paper's tables)
└── requirements.txt
```

## 3. Quick start

```bash
pip install -r requirements.txt

# (optional) explicitly (re)generate the synthetic dataset
python scripts/generate_synthetic_data.py --n-vessels 60 --points-per-vessel 144

# run the full pipeline: ingestion -> ... -> anomaly score -> evaluation -> figures
python scripts/run_pipeline.py --n-vessels 60 --points-per-vessel 144 --epochs 100
```

This writes:
- `outputs/checkpoints/nusantara_ais_best.pt` — best model checkpoint (resumable, contains config + optimizer state)
- `outputs/reports/experiment_report.json` — every metric (detection metrics, per-class scores, false-positive/failure
  analysis, ARI sensitivity, baseline comparison, runtime/complexity)
- `outputs/figures/*.png` — dataset statistics, score distributions, ROC/PR curves, embedding projection, attention
  summary, ablation comparison

To score new AIS data with a trained checkpoint:

```python
from nusantara_ais.config import Config
from nusantara_ais.data.ingestion import DataIngestor
from nusantara_ais.inference.infer import score_new_data

cfg = Config()
dataset = DataIngestor(cfg).load()
report = score_new_data(new_raw_ais_df, dataset, cfg, "outputs/checkpoints/nusantara_ais_best.pt")
# report: mmsi, timestamp, ari, anomaly_score, risk_tier, feat/struct/unreliability percentiles
```

Run the test suite:

```bash
python -m pytest tests/ -v
# or, dependency-free:
python tests/test_core.py
```

## 4. Using real datasets instead of synthetic

Drop the following files (parquet or csv) into `data/raw/`, matching the
schemas documented in `src/nusantara_ais/data/ingestion.py`:

| File | Source | Required columns |
|---|---|---|
| `ais_raw` | Global Fishing Watch | `mmsi, timestamp, lat, lon, sog, cog, heading, nav_status, vessel_type, length, width, draft` |
| `ports` | Indonesian Ports | `port_id, lat, lon, capacity_teu` |
| `shipping_lanes` | BIG National Shipping Lane Network | `lane_id, segment_id, lat_start, lon_start, lat_end, lon_end` |
| `coastline` | Coastline | `lat, lon, seq` |
| `bathymetry` | Bathymetry | `lat, lon, depth_m` |
| `bmkg_weather` | BMKG (wave/wind/weather) | `station_lat, station_lon, timestamp, wind_speed_ms, wave_height_m, wave_direction_deg, weather_code` |
| `protected_areas` | Marine protected areas | `area_id, lat_center, lon_center, radius_km` |

No other file needs to change — `DataIngestor` transparently prefers real
data over synthetic per-table, so you can mix (e.g. real AIS + synthetic
BMKG while waiting on a data-sharing agreement).

Note on terminology: shipping-lane infrastructure is referred to throughout
the codebase and outputs strictly as **National Shipping Lane** / **Shipping
Lane Network** (BIG spatial-planning data), consistent with this project's
defined dataset scope.

## 5. The three contributions

### 5.1 Physics-informed AIS Reliability Index (ARI) — Contribution 1

`src/nusantara_ais/features/reliability.py`

ARI ∈ [0, 1] is a convex combination of five physically-grounded sub-scores:

| Sub-score | Physical basis | Weight |
|---|---|---|
| Kinematic plausibility | acceleration/turn-rate vs. real-world vessel limits | 0.30 |
| Coordinate-jump | implied speed between fixes vs. the vessel's own recent max SOG | 0.20 |
| AIS gap ratio | fraction of expected messages actually observed | 0.20 |
| Blackout ratio | fraction of elapsed time spent in a >30min silent gap | 0.15 |
| Spoofing indicator | consistency of static identity fields (type/length/width/draft) over time | 0.15 |

A convex (not multiplicative) combination is used deliberately, so no single
noisy sub-score collapses the whole index, while the weighting still
emphasizes the strongest physical evidence (kinematics + jump = 50%).
Full equations and rationale are in the module docstring.

### 5.2 Spatially-aware heterogeneous maritime graph — Contribution 2

`src/nusantara_ais/graph/{nodes,edges,builder}.py`

4 node types (`vessel_state`, `port`, `shipping_lane_segment`, `grid_cell`),
7 forward edge types + mirrored reverse edges, all with justified,
distance-based (`exp(-d/radius)`) or temporal-decay (`exp(-dt/tau)`)
weighting. The graph is generated as a sequence of overlapping
**temporal snapshots** (not one static graph), bounding memory per-snapshot
and enabling PyG's native heterogeneous batching. See the module docstrings
for the full mathematical justification of every edge type, including why
`grid_cell` nodes are the mechanism that makes the graph genuinely
"spatially-aware" (2-hop vessel-to-vessel spatial context beyond the direct
proximity-edge budget).

### 5.3 Context-aware Maritime Anomaly Score — Contribution 3

`src/nusantara_ais/models/anomaly_score.py`

```
AnomalyScore(i) = 0.5 * percentile_rank(feature_reconstruction_error_i)
                + 0.3 * percentile_rank(structure_reconstruction_error_i)
                + 0.2 * percentile_rank(1 - ARI_i)
```

Percentile-rank normalization makes the score robust to the differing raw
scales of each signal without hand-tuned constants. The weights are the
subject of the ARI-sensitivity experiment (`evaluation/ablation.py::ari_sensitivity_analysis`).
`risk_tier()` maps the continuous score onto LOW/MODERATE/HIGH/CRITICAL
operational tiers (quartile-based) **without** ever collapsing back into a
binary illegal-fishing classification.

## 6. Model architecture

```
vessel_state.x [N_v, ~44]  ─┐
port.x [N_p, 3]             ├─► per-type Linear input projection ─► HetGAT (L=2 layers, H=4 heads) ─► z_HetGAT[vessel_state] ∈ R^[N_v, 128]
lane.x [N_l, 5]             │
grid_cell.x [N_g, 4]       ─┘
                                                                              │
                                                                              ▼
                                                        GraphAutoEncoder: Linear -> z ∈ R^[N_v, 32]
                                                          ├─ Feature decoder  -> x_hat  (MSE loss)
                                                          ├─ Structure decoder -> sigmoid(z_i·z_j) (BCE loss, neg. sampling)
                                                          └─ ARI head         -> ari_hat (MSE loss, auxiliary regularizer)
```

Every tensor dimension, forward pass, loss term, and design decision is
documented inline in `hetgat.py`, `gae.py`, and `anomaly_score.py`.

## 7. Evaluation suite

All implemented in `evaluation/{metrics,ablation,visualization}.py` and run
automatically by `scripts/run_pipeline.py`:

- Dataset statistics (SOG/ARI distributions, class balance, spatial footprint)
- Baseline comparison: Isolation Forest, One-Class SVM, XGBoost (supervised
  ceiling), GraphSAGE, GAT, full HetGAT+GAE — `evaluation/ablation.py::run_baseline_comparison`
- Ablation study: no-structure-loss, no-ARI-loss, no-heterogeneity variants
  — `run_ablation_study`
- Feature importance via permutation — `feature_importance_via_permutation`
- ARI sensitivity (score-weight sweep) — `ari_sensitivity_analysis`
- Embedding analysis (deterministic PCA projection) — `embedding_analysis`
- Attention analysis (per-relation mean HetGAT attention) — `attention_analysis`
- Case study (per-vessel score-over-time + trajectory map) — `visualization.plot_case_study`
- False-positive / failure analysis by behavior class — `metrics.false_positive_analysis`, `metrics.failure_analysis`
- Computational complexity (theoretical, documented in `metrics.runtime_and_complexity`) + empirical runtime profiling

Detection quality against synthetic ground truth is measured with
threshold-free ranking metrics (ROC-AUC, PR-AUC) since the model outputs a
continuous score, never a hard classification.

## 8. Reproducibility

- `utils/reproducibility.set_global_seed` fixes Python/NumPy/PyTorch
  CPU+CUDA RNGs, `PYTHONHASHSEED`, and forces deterministic cuDNN kernels.
- Every checkpoint stores the full serialized config, optimizer state, and
  epoch, enabling exact resume.
- Snapshot train/val/test splits are **chronological**, not random, which is
  the only leak-free choice given `temporal_next`/`spatial_proximity` edges
  connect adjacent-in-time snapshots.
- `Config.save()` / `Config.load()` round-trip the entire experiment
  configuration to/from YAML.

## 9. Known limitations / next steps for the full paper submission

- The synthetic data generator (`data/synthetic.py`) is a physically
  realistic stand-in for the licensed real datasets (GFW/BMKG/BIG), used so
  the full pipeline is runnable and testable in this environment. Swapping
  in real data is a `data/raw/` drop-in (Section 4) — no other code changes.
- Default hyperparameters (`config.py`) are reasonable starting points, not
  yet tuned; `ari_sensitivity_analysis` and `run_ablation_study` are the
  intended tools for that tuning pass once real data is available.
- `historical.py`'s per-vessel rolling-window computation is a clear,
  auditable Python loop rather than a fully vectorized implementation; for
  fleet sizes beyond a few hundred vessels at full temporal resolution, this
  should be swapped for a vectorized/Numba implementation before a
  production run — the module docstring flags this explicitly.

# Phase 24 gate — is a station-graph GNN scientifically justified? (stage 39, D13)

| | |
| --- | --- |
| **Date** | 2026-10-11 |
| **Script** | `scripts/phase24_gate.py` → `reports/phase24_graph_gate.log` (exit 0), `phase24_graph_gate.json`, `phase24_graph_edges.csv`, `figures/phase24_gate.png` |
| **Inputs** | `data/processed/static_vectors.csv` (133 stations: lat/lon, river, basin, elevation) · `data/processed/experimental_dataset.csv` (`wl_pct_danger`, stored `up_wl_pct_danger_lag{1,2}`) · `data/processed/sequence_index.csv` (temporal coverage) · Phase-11C upstream rule |
| **Checks** | **G1–G3 PASS** (station master; series coverage; edge identity reproduces the stored Phase-11C feature) |
| **Verdict** | **PASS (narrow, pre-registered).** A defensible hydrology-based edge set exists and neighbours are demonstrably dependent; the GNN earns one registered attempt **only** on the P3 unseen-station protocol, and only if it beats the temporal-only LSTM there. |

D13: GNN (Phase 24) runs only if this gate passes; **a documented negative result is an
acceptable outcome**. This report is 24.1; 24.2 (the models) runs only on a pass. No model
was trained here and the frozen test was not read.

---

## 1. The gate question

The workflow asks three things with numbers: (a) can a defensible edge set be built
(**river-network / basin first, distance only as a documented fallback**); (b) what is the
graph coverage / connectivity / edge overlap / missingness; (c) do stations actually carry
information about one another (upstream level → downstream at plausible lags). Prior
evidence to weigh: the Phase-11C `up_*` columns score strongly *individually* (univariate
oriented AUC **0.8915** for `up_wl_pct_danger_lag1`, Phase 13 L6) yet the Phase-16 **F14-2**
ablation found adding them to the LINEAR baseline moved val-2022 PR-AUC by **+0.0002** — i.e.
neighbour signal exists but looked redundant with the station's own state. The gate must
decide whether that redundancy closes the stage or merely sets expectations.

## 2. Candidate edge sets (priority order)

| graph | definition | kind |
| --- | --- | --- |
| **river_upstream** *(primary)* | Phase-11C rule: same `river` string, neighbour ≥ **0.5 m** higher elevation, nearest by haversine, ≤ **250 km** → directed upstream → downstream | hydrological |
| same_river_250km | all same-river pairs ≤ 250 km | hydrological (looser) |
| basin_clique | all pairs sharing a basin (5 basins) | administrative/coarse |
| knn3 / knn5 | nearest-*k* gauges by haversine | distance fallback |

The primary graph is **not** distance-based; distance is used only to make the graph
connected (§8).

## 3. Coverage & connectivity

| graph | edges | stations covered / 133 | isolated | components | largest | connected |
| --- | ---: | ---: | ---: | ---: | ---: | :--: |
| **river_upstream** | 52 | **82 (62 %)** | 51 | 81 | 10 | no |
| same_river_250km | 111 | 83 | 50 | 80 | 10 | no |
| basin_clique | 2 178 | 133 | 0 | 5 | 50 | no |
| knn3 | 260 | 133 | 0 | 2 | 128 | no |
| knn5 | 415 | 133 | 0 | **1** | **133** | **yes** |

52 directed river edges (44 stations acting as someone's upstream, 52 as a downstream). The
hydrological graph is **sparse and fragmented** (81 components, largest 10; mean degree 0.78)
— it is a set of small same-river chains, not one delta-wide network. **51 of 133 stations
(38 %) are isolated** and need the distance fallback. `knn5` is the smallest distance graph
that is fully connected; `basin_clique` covers everything but is a coarse complete graph
(mean degree 33) that would let message passing average across unrelated hydrology.

## 4. Edge identity validated against the stored feature

Re-running the Phase-11C rule on the 133 model-ready stations yields 52 edges; **50 of them
reproduce the stored `up_wl_pct_danger_lag1` column exactly** (upstream `wl_pct_danger` at
*t−1*, match ≥ 0.99 on every overlapping day — mean match **0.9997**), and exactly **50
stations carry an `up_*` value**, matching Phase 11C's "50 in the model-ready table". The two
extra reconstructed edges touch stations with no stored upstream (no partner with data). The
reconstructed graph is therefore the *same mapping* the project already trusts (G3 PASS).

## 5. Temporal overlap & missingness (river edges)

Per-edge overlap of the two endpoints' WL records: **min 69 d, median 1 226 d, max 2 062 d**;
Jaccard overlap median **0.677** (per-edge missingness median 0.32). **10 / 52 edges have
< 1 year of joint history**. Records span 2017-01-06 → 2026-10-05; the network grows over
time (85 → 133 stations, risk R7), so early-overlap is the binding constraint for a few edges
— an argument for masking (as Dataset B already does) rather than imputation.

## 6. Cross-station dependence (the core evidence)

Mean Pearson correlation between **upstream WL(t−lag)** and **downstream WL(t)**, over the 50
data-bearing river edges, versus a null of 4 000 random non-edge station pairs:

| lag (days) | river edges (mean / median) | null (mean) | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.900 / 0.958 | 0.578 | **+0.322** |
| 1 | 0.892 / 0.953 | 0.574 | **+0.317** |
| 2 | 0.863 / 0.940 | 0.568 | +0.296 |
| 3 | 0.832 / 0.916 | 0.559 | +0.273 |

**Every** river edge has lag-1 r ≥ 0.5. River neighbours move almost in lockstep (median
r ≈ 0.95) and the correlation decays with lag exactly as propagation predicts. Note the null
is *not* zero (mean 0.58): Bangladeshi gauges share a strong monsoon seasonal cycle, so about
half the raw correlation is common season — the **river-specific excess (+0.27 to +0.32)** is
the part attributable to hydrological coupling. `figures/phase24_gate.png` (left panel).

**Conclusion for (c): yes — stations carry information about one another, strongly and with a
plausible lag structure.**

## 7. Does that signal add *predictive* value? (and the F14-2 contrast)

On the 50-station / 59 463-row subset that has an upstream, a plain logistic model was fit
(train < 2022, evaluated on val-2022 with the project median-imputer):

| target | own state only | own + upstream lag{1,2} | Δ PR-AUC |
| --- | ---: | ---: | ---: |
| `label_24h` | 0.9267 | 0.9293 | **+0.0026** |
| `label_48h` | 0.9017 | 0.9106 | **+0.0089** |
| `label_72h` | 0.8774 | 0.8891 | **+0.0117** |

Against a *minimal* own baseline the neighbour adds a small, **horizon-increasing** increment
(+0.003 → +0.012). This is consistent with propagation and is *not* contradicted by F14-2,
which added `up_*` to the **full engineered LINEAR set** (35 collinear columns, VIF-pruned)
and found +0.0002 — there the neighbour information was already carried by the station's own
lagged/rolling hydrology and by the `nb*_` neighbourhood columns. Read together: **relational
information is real, but mostly redundant with same-station persistence on P1; its distinct
value can only be tested where same-station persistence does not transfer — unseen stations.**

## 8. Verdict and the pre-registered plan

**PASS (narrow).** All three gate criteria are met: a defensible hydrology-first edge set
exists and reproduces the project's own upstream mapping; dependence is strong with a
propagation shape; and the neighbour carries a small positive increment. The gate does **not**
claim the GNN will win — it authorises exactly one fair test under pre-registered kill
criteria.

**Approved graph for 24.2.** `G = river_upstream edges (primary, hydrological) ∪ knn5
(distance fallback, to connect the 51 isolated nodes)`. Both a hydrology-only variant and the
union variant are reported; distance-only is never the sole graph. The edge list is frozen as
an artifact with a manifest + SHA-256 before any model sees it.

**Candidates.** `L4-01 GCN` and one attention/temporal variant — `L4-02 GAT` (attention over
neighbours) or `L4-04 GNN→LSTM` — node features = per-station daily variables, temporal
encoder as in Phase 21 (L\*=3). Same splits, metrics, seed and budget (L4 = 20 configs, D14);
same imbalance rule as every family.

**Pre-registered acceptance / kill criteria.**
1. Primary: PR-AUC on **P3 spatial CV (unseen stations)** vs the temporal-only LSTM
   (`L3-01`) at 24/48/72 h, with a paired bootstrap CI. *P3 is the only protocol where a graph
   model has a hypothesis the per-node model cannot have.*
2. Secondary: P1 `val_2022` (expected ~equal — F14-2).
3. **If no meaningful P3 gain (CI spanning 0) over the temporal-only LSTM, stage 39 closes
   as a documented rejection** and no Transformer-on-graph is attempted (Phase 25 stays
   temporal-only or is likewise rejected).

## 9. Registry & handoff

* Gate `phase24_graph_gate` (attached to L4-01/L4-02/L4-04 in `phase17_experiment_registry.csv`)
  is **discharged by this report**. The framework still marks those rows `gated`; 24.2 will
  amend them `gated → planned` and re-run `scripts/phase17_framework.py` (its F1m check is
  updated at that point). L4-03 `transformer_temporal` is gated by `phase25_transformer_gate`
  and is **unaffected**.
* New data artifact required by 24.2: the station graph (`data/processed/station_graph.*` +
  manifest/SHA), authority `added_phase24_gate` (expected `+= 52` river + fallback edges).
* Reuse: the frozen `up_wl_pct_danger_lag{1,2}` columns already encode a single fixed
  upstream edge — the GNN must beat *that*, not merely reproduce it.

**Decision: GATE PASSED (narrow) — proceed to 24.2 under the criteria above.**

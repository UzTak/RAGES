# RAGES+ Development Plan

This document summarizes the development procedure for the bootstrapped, bottom-up RAGES+ training pipeline. The goal is to implement the pipeline in a way that cleanly separates physical/optimization feasibility, intent-dependent preference, and parser-side language handling.

IMPORTANT: this is a research codebase for prototyping, so simplicity / readability of the codebase is critical, rather than a lot of assertions. Prioritize the simplicity of the overall code organization.

## Important: Overall contribution of the paper

1. Development of the intermediate representation (IR) and decomposition (behavior graph and waypoint constriants) in order to make a intent (free-form text)-to-trajectory pipeline 

2. Scalable dataset generation in the dataset-limited world: behavior-graph walking based sampling technique 

3. Demonstration that the proposed method can (i) trace the mission designer's intent and better than other heuristic methods. 


## 0. Core design decision

Use a **sigma-free verifier surrogate**:

```text
Q_psi(s, a) -> preference-free verifier metrics
rho_sigma(Q_psi(s, a)) -> intent/preference-dependent scalar score
```

where

```text
s = (x0, z)
z = (oe(t0), r_KOZ, beta)
a = (b_seq, t_f)
```

and

```text
Q_psi(s, a) = [p_conv, E[m_1 | conv], ..., E[m_|M| | conv], optional quantiles]
```

The preference vector `sigma` must **not** be an input to `Q_psi` unless it changes the actual verifier outcome. It should only affect:

1. hard candidate filtering / masking, and
2. external scalar scoring through `rho_sigma`.

This keeps `Q_psi` as a preference-free approximation of the frozen verifier stack.

## 1. Authority hierarchy

The implementation should preserve the following hierarchy:

```text
SCP certifies.
Q ranks.
pi proposes.
Mask contains.
Parser translates.
```

Meaning:

- SCP is the final authority for feasibility and trajectory quality.
- `Q_psi` predicts verifier outcomes but does not certify them.
- The policy `pi_theta` proposes structured behavior candidates.
- Token/action masks enforce grammar, precedence, windows, and hard admissibility.
- The parser is a separate text-to-IR component and is not part of the learned guidance stack.

## 1.1 Current repository baseline

The current repository already implements the RAGES v1 pipeline:

```text
reasoning LoRA -> waypoint model -> SCP
```

The implementation entrypoint is:

```text
src/rages.py
```

Current artifacts are stored under:

```text
model/reasoning_model/
model/wyp_model/
data/wyp_data/
data/reasoning_data/
```

Reusable v1 components:

- behavior vocabulary and campaign graph: `src/parameters.py`,
- graph and scenario sampling utilities: `work/datagen_wyp.py`,
- explicit RAGES+ action schema and deterministic sampler: `src/parameters.py`, `src/rages_sampling.py`,
- waypoint model library (`p_phi` config, featurization, GMM/VAE classes, decoding, checkpoint I/O, inference): `src/wyp_predictor.py`,
- waypoint model baseline training script: `work/train_wyp_predictor.py`,
- frozen-waypoint candidate rollout and SCP metric computation: `work/datagen_reasoning.py`,
- reasoning SFT baseline: `work/train_reasoning_model.py`,
- end-to-end smoke wrapper: `src/rages.py`.

RAGES+ should reuse these components, but it should introduce explicit library-level interfaces for `Scenario`, `IR`, `Action`, `WaypointPlan`, `VerifierResult`, `QOutput`, and artifact versions. The current scripts can remain as v1 baselines, while new RAGES+ modules should avoid depending on ad hoc JSON rows or script-local globals.

### 1.2 Action schema contract

The inference-time action is strictly:

```text
a = (b_seq, tof_steps)
```

implemented as `Action` in `src/parameters.py`.

Do **not** include `policy`, `dt_orbits`, transfer windows, or target-domain traces in the action. Those fields are data-curation metadata used to generate scalable supervision without expert labels. They may be stored with training samples for auditability and regeneration, but the policy and Q model should treat the action only as `(b_seq, tof_steps)`.

Current separation:

```text
Action:
  b_seq
  tof_steps

ActionCurationMetadata:
  policy
  dt_orbits
  dt_ranges
  target_domains
```

This distinction matters because campaign `policy` is a curation mechanism, not an inference-time decision variable. At deployment, RAGES+ should not require selecting or exposing the campaign policy that happened to generate a training sample.

## 2. Stage 0 — Samplers

### 2.1 Scenario sampler

Implement a scenario sampler that produces:

```text
s = (x0, z)
z = (oe(t0), r_KOZ, beta)
```

Required outputs:

- initial relative/orbital state `x0`,
- orbital elements or environment parameters `oe(t0)`,
- keep-out-zone radius or geometry `r_KOZ`,
- scenario/domain parameter `beta`.

The sampler should support deterministic seeding and train/validation/test splits.

### 2.2 IR sampler

Implement an intent-representation sampler that produces synthetic IR tuples:

```text
IR = (sigma, dz, g)
```

where:

- `sigma` encodes preferences, priorities, or thresholds,
- `dz` encodes desired terminal/domain changes,
- `g` encodes goal/task class information.

The IR sampler should cover:

- priority permutations,
- threshold settings,
- hard filters,
- terminal domains,
- command-class taxonomy.

Important invariant:

```text
The learned guidance policy never sees raw text.
Raw text is handled only by the separate parser workstream.
```

## 3. Hard filters versus soft preferences

Separate IR-derived information into two categories.

### 3.1 Hard filters

Hard filters prune the candidate set before policy/Q ranking. Examples:

- invalid behavior grammar,
- precedence violations,
- forbidden behavior windows,
- impossible terminal domains,
- hard mission budgets,
- discrete safety prohibitions.

These define the admissible action set:

```text
A_mask(s, IR)
```

They do not enter `Q_psi`.

Current status (v0): the IR-derived hard filter is a pass-all placeholder
(`hard_filter_pass_all` in `src/rages_scoring.py`). Grammar, precedence, and
window validity are already enforced by the behavior-graph sampler / action
mask, so the v0 filter admits every mask-valid candidate. IR-derived hard
constraints (forbidden windows, budgets, terminal-domain prohibitions) plug in
later by replacing this function without touching the ranking path.

### 3.2 Soft preferences

Soft preferences affect ranking but not physical feasibility. Examples:

- fuel preference,
- time preference,
- safety-margin preference,
- terminal-accuracy preference,
- risk tolerance.

These enter only through the external scalar score:

```text
score = rho_sigma(Q_psi(s, a))
```

They should not be embedded inside the output definition of `Q_psi`.

## 4. Stage 1 — Waypoint model `p_phi`

### 4.1 Purpose

Train a waypoint model that realizes a behavior-level action into waypoint or constraint parameters used by SCP.

Expected interface:

```text
w_hat = p_phi(s, a)
a = (b_seq, t_f)
```

where `w_hat` denotes waypoint constraints, passage constraints, or other SCP initialization/constraint parameters.

### 4.1.1 Repository baseline and target model class

The current repository baseline is the conditional GMM/VAE waypoint filler. The model library (config, featurization, model classes, `constrained_fill` decoding, checkpoint loading, and `predict_wyp_seq` inference) lives in:

```text
src/wyp_predictor.py
```

with the training loop kept as the script:

```text
work/train_wyp_predictor.py
```

with rollout data generated by:

```text
work/datagen_wyp.py
```

This baseline should be retained for comparison and smoke testing. For RAGES+, the preferred waypoint model should be a **conditional flow-matching model** because waypoint placement is naturally multimodal: the same `(s, a)` can admit several geometrically distinct waypoint plans that are all SCP-feasible but differ in fuel, observation quality, safety margin, or terminal conservatism.

Recommended RAGES+ waypoint model:

```text
v_theta(y_t, t, s, a) -> dy_t / dt
y = [x_seq, dt_seq]
```

This is implemented as `ConditionalFlowMatcher` in `src/wyp_predictor.py`,
following the same `forward` / `compute_loss` / `sample_y` protocol as the
GMM/VAE baselines (training script pending, item 8).

where:

- `x_seq` contains phase waypoint states,
- `dt_seq` contains positive transfer-time fractions constrained to the simplex,
- conditioning uses the same fields as the current waypoint baseline: `x0`, `tof`, `oec0_modified`, `artms_scale_range_1e3`, `koz_dim`, and `b_seq`.

Initial implementation should support:

- conditional flow matching over normalized waypoint states,
- a simplex-safe representation for `dt_seq` such as logits followed by masked softmax,
- reward or quality weighting using SCP rollout metrics,
- multiple samples per `(s, a)` during analysis,
- deterministic ODE decoding for downstream Q-label generation.

The existing GMM model is still useful as:

- the v1 baseline,
- a fast smoke-test model,
- an ablation against flow matching,
- a fallback if flow-matching training is not yet available.

### 4.2 Training data

Use SCP rollouts as supervision. Each training record should include:

```text
(s, a, w, metrics, converged_flag)
```

where:

- `s` is the scenario,
- `a` is the behavior sequence and time of flight,
- `w` is the waypoint/constraint representation,
- `metrics` are preference-free verifier metrics,
- `converged_flag` indicates SCP success.

### 4.3 Loss

For the current GMM baseline, use reward-weighted MLE:

```text
L_phi = - E[omega(tau) log p_phi(w | s, a)]
```

The weighting should be clipped or temperature-smoothed to avoid support collapse.

For the RAGES+ flow-matching model, use a conditional flow-matching objective:

```text
L_phi = E[t, y_0, y_1, s, a, omega]
        omega * || v_theta(y_t, t, s, a) - u_t(y_0, y_1) ||^2
```

where:

- `y_1` is an SCP-supervised waypoint plan,
- `y_0` is the base noise sample,
- `y_t` is the interpolated state,
- `u_t` is the target conditional velocity,
- `omega` is a clipped or temperature-smoothed quality weight.

The weighting must not collapse the support to a single mode. Evaluation should explicitly check sample diversity and downstream SCP success, not only mean error.

### 4.4 Deterministic decoding rule

Define a deterministic decoding rule for `p_phi` before training `Q_psi`.

Recommended main-path rule:

```text
w_hat = deterministic_decode(p_phi(s, a))
```

Examples:

- deterministic ODE solve from a fixed base sample for flow matching,
- mode prediction for the GMM baseline,
- mean prediction only as a baseline or smoke-test path,
- fixed decoding seed,
- fixed top-k/top-p rule if stochastic decoding is unavoidable.

This is necessary because `Q_psi(s, a)` must approximate a well-defined frozen verifier outcome. If `p_phi` is stochastic and uncontrolled, Q labels become ambiguous.

For RAGES+, avoid using a simple mean waypoint as the main path. Mean decoding can blur distinct feasible modes into an infeasible or low-quality waypoint plan. The main deterministic path should be a fixed ODE decode or a documented mode-selection rule.

### 4.5 Freeze condition

After Stage 1:

```text
Freeze p_phi.
All downstream labels must be generated using this frozen p_phi version.
```

If `p_phi` changes later, all Stage 2 and above data must be regenerated or invalidated.

## 5. Stage 2 — Verifier surrogate `Q_psi(s, a)`

### 5.1 Purpose

Train a sigma-free, preference-free approximation of the frozen verifier stack:

```text
frozen p_phi + SCP verifier
```

The verifier call is:

```text
w_hat = deterministic_decode(p_phi(s, a))
result = SCP(s, w_hat)
```

The Q model predicts the outcome of this call.

### 5.2 Q output definition

Recommended output:

```text
Q_psi(s, a) = {
    p_conv,
    metric_means,
    optional_metric_quantiles
}
```

where:

```text
p_conv = P(SCP converges | s, a)
metric_means[j] = E[m_j | s, a, SCP converged]
```

Optional quantile heads are useful when the score needs lower-confidence-bound ranking:

```text
q_alpha[j] = alpha-quantile of metric m_j conditioned on convergence
```

Do not include `sigma` in the Q input unless `sigma` changes the actual verifier trajectory distribution or SCP problem definition.

### 5.3 Dataset generation

Generate many candidates for each scenario:

```text
for s in scenarios:
    for a in candidate_actions(s):
        w_hat = p_phi(s, a)
        result = SCP(s, w_hat)
        store(s, a, converged_flag, metrics_if_converged)
```

Retain all candidates:

```text
Keep winners, losers, failed samples, near misses, and dominated candidates.
```

Do not perform argmax-only distillation. The ranking model needs contrastive information.

### 5.4 Censoring-aware training

Use two types of heads:

1. convergence head trained on all samples,
2. metric heads trained only on converged samples.

Loss structure:

```text
L_Q = BCE(p_conv, converged_flag)
    + 1[converged] * L_metric(metric_means, metrics)
    + optional quantile losses
```

The metric heads should not be trained on failed SCP samples unless a well-defined failure metric is explicitly introduced.

### 5.5 Calibration and ranking gates

Before freezing `Q_psi`, evaluate it on cached true-SCP fixtures.

Required gates:

- ECE / calibration error for `p_conv`,
- Brier score or NLL for convergence probability,
- Kendall-tau ranking within matched candidate groups,
- top-1 regret against true-SCP ranking,
- false-positive rate for candidates predicted feasible but SCP-failing,
- quantile calibration if quantile heads are used.

Important:

```text
Kendall-tau must be computed within candidate groups sharing the same scenario and IR/preference context.
Global ranking across unrelated scenarios is not meaningful.
```

### 5.6 Freeze condition

After Stage 2:

```text
Freeze Q_psi for the next policy-training phase.
```

If Q is refit during the drift loop, the affected policy-training phase must be restarted or resumed with explicit version tracking.

## 6. External scoring `rho_sigma`

### 6.1 Purpose

Convert preference-free Q outputs into a scalar score for candidate ranking.

Example:

```text
score = rho_sigma(Q_psi(s, a))
```

A typical form is:

```text
score = utility_sigma(metric_means)
      - lambda_fail * (1 - p_conv)
      - optional uncertainty penalty
```

or, for conservative ranking:

```text
score = LCB_sigma(Q_psi(s, a))
```

### 6.2 Invariant

`rho_sigma` is the only place where soft preference scalarization should occur.

```text
Q predicts what happens.
rho_sigma decides what is preferred.
```

This separation is central to the method.

### 6.3 Initial realization (v0): tolerance-based lexicographic ranking

The first `rho_sigma` is an ordering, not a scalar score. This is sufficient
because every downstream consumer (group-relative GRPO advantages, top-1
selection, within-group Kendall tau) only needs ranks. Implementation:
`src/rages_scoring.py`.

Definition: metrics are compared in intent-priority order; two candidates are
tied on metric `m_k` when

```text
|m_k(i) - m_k(j)| <= eps_k
```

and the comparison falls through to the next-priority metric.

Implementation detail: pairwise tolerance ties are not transitive, so each
metric is instead quantized into eps-width buckets,

```text
key_k = floor(m_k / eps_k)        (sign-flipped for max-metrics)
```

and candidates are sorted by the lexicographic key tuple. This keeps the
order total, deterministic, and per-candidate (each key is computable
independently of the candidate group, matching the per-candidate
`rho_sigma(Q_psi(s, a))` interface). Two values within `eps_k` of each other
can still straddle a bucket boundary; the deviation from the pairwise
definition is bounded to one bucket. `eps_k = 0` gives strict comparison and
recovers the v1 `rank_det` behavior in `work/datagen_reasoning.py`.

Sigma parameterization under this realization:

```text
sigma = (priority permutation, eps vector, p_conv feasibility threshold)
```

All three are samplable by the IR sampler (priority permutations and
threshold settings, cf. Sec. 2.2). Feasibility is the first lexicographic
key: with true-SCP labels it is the converged flag; with Q outputs it is
`p_conv >= threshold`, where the threshold is a sigma-level risk-tolerance
parameter. Infeasible candidates rank after all feasible ones, and candidates
with non-finite metrics rank after all fully valid ones.

The default eps values in `rages_scoring.DEFAULT_EPSILONS` are placeholders;
once the Stage 2 candidate dataset exists, recalibrate them per metric as a
fraction of the median within-scenario candidate spread
(`epsilons_from_metric_spread`).

Scalar forms of `rho_sigma` (weighted utility, LCB) remain available as later
variants and as the scalarization baseline in the main experiments (Sec. 13);
they do not change the Q interface.

## 7. Stage 3a — Format SFT

### 7.1 Purpose

Train a structured policy checkpoint that knows how to emit valid behavior-level actions under the mask.

Expected policy interface:

```text
pi_theta(a | s, IR, mask)
```

where:

```text
a = (b_seq, t_f)
```

The output should be structured, not free-form text.

### 7.2 Training target

Use IR-conditioned structured outputs:

```text
behavior_sequence: [b_1, b_2, ..., b_K]
time_of_flight_bin: k_tf
optional_constraint_flags: [...]
```

Use masked decoding from the start.

### 7.3 Data

Include:

- successful behavior examples,
- failure-context examples,
- executive re-query examples,
- examples near mask/filter boundaries.

### 7.4 Reasoning traces

Do not make free-form chain-of-thought traces part of the main method. If interpretability traces are useful, treat them as an ablation or auxiliary output.

Main method:

```text
format-only structured SFT
```

### 7.5 Freeze condition

After Stage 3a:

```text
Use this checkpoint as the KL anchor for Stage 3b.
```

## 8. Stage 3b — GRPO / verifier-proxy-guided policy improvement

### 8.1 Purpose

Improve the structured behavior proposal policy using Q-based group-relative ranking while maintaining grammar validity through masks.

This stage is contribution-bearing but should be framed as:

```text
verifier-proxy-guided structured policy improvement
```

not merely as application of GRPO.

### 8.2 Sampling

For each sampled prompt/context:

```text
input = (s, IR)
mask = build_mask(s, IR)
{a_i}_{i=1}^G ~ pi_theta(. | s, IR, mask)
```

All sampled actions must satisfy the token/action mask.

### 8.3 Reward assignment

For each candidate:

```text
q_i = Q_psi(s, a_i)
r_i = rho_sigma(q_i)
```

Use rank-based or group-relative advantage:

```text
A_i = rank_or_group_advantage(r_i within group)
```

No learned critic is introduced. `Q_psi` is a verifier proxy used for reward assignment, not a baseline critic.

### 8.4 KL anchor

Maintain a KL penalty to the Stage 3a checkpoint:

```text
L = L_GRPO + beta_KL * KL(pi_theta || pi_3a)
```

This prevents the policy from drifting away from valid structured behavior syntax.

### 8.5 Proxy-drift audit

Every `k` updates, run a true-SCP audit:

```text
sample policy candidates
score by Q
execute frozen p_phi + SCP
compare predicted ranking/quality against true verifier outcome
```

Log:

- predicted score,
- true SCP convergence,
- true metrics,
- rank mismatch,
- top-1 regret,
- false-positive rate,
- proxy exploitation examples.

## 9. Stage 4 — Drift loop

### 9.1 Trigger

Trigger the drift loop if audit performance degrades, for example:

- high false-positive rate,
- worsening top-1 regret,
- poor rank correlation,
- repeated Q exploitation by the policy,
- large mismatch between predicted and true SCP metrics.

### 9.2 Relabeling

Collect on-policy candidates from the current policy:

```text
(s, IR, a) ~ current pi_theta
```

Run true verification:

```text
w_hat = p_phi(s, a)
result = SCP(s, w_hat)
```

Append relabeled samples to the Stage 2 dataset.

### 9.3 Refit Q

Refit or fine-tune `Q_psi` using the expanded dataset.

Version the new model explicitly:

```text
Q_v1, Q_v2, ...
```

### 9.4 Resume policy improvement

Resume Stage 3b using the updated Q model. Record that the reward model changed.

Updated invariant:

```text
Each lower layer is frozen during a given upper-layer optimization phase.
Audit-triggered lower-layer updates invalidate or version the affected upper-layer phase.
```

## 10. Parallel parser workstream

The parser is separate from the learned guidance stack.

### 10.1 Parser interface

```text
raw_text -> IR = (sigma, dz, g, filters)
```

Recommended implementation:

- frozen frontier LLM,
- few-shot examples,
- schema-constrained decoding,
- abstention when uncertain.

### 10.2 Parser benchmark

Evaluate offline using:

- slot-level F1,
- command-class accuracy,
- abstain precision,
- abstain recall,
- per-taxonomy performance,
- malformed-command rejection rate.

The parser should not be coupled to Stage 1--4 training.

## 11. Dataset/versioning rules

Use strict artifact versioning.

Recommended versions to store:

```text
sampler_version
behavior_vocab_version
mask_version
p_phi_version
Q_version
rho_sigma_version
pi_3a_version
pi_3b_version
SCP_solver_version
SCP_config_hash
```

Critical invalidation rules:

1. If `p_phi` changes, regenerate Stage 2 labels.
2. If the SCP formulation changes, regenerate Stage 2 labels.
3. If the behavior vocabulary changes, regenerate Stage 1--3 data.
4. If hard masks change, regenerate Stage 3a/3b training data or clearly version the mask.
5. If `rho_sigma` changes, Stage 2 does not need retraining, but Stage 3b reward logs must be versioned.
6. If only soft preference scalarization changes, keep Q fixed and re-run ranking/policy training as needed.

## 12. Minimum implementation order

Recommended development order:

1. Done: promote the existing behavior vocabulary and campaign graph in `src/parameters.py` into an explicit action schema. `Action` contains only `b_seq` and `tof_steps`; curation-only fields live in `ActionCurationMetadata`.
2. Done: promote the existing scenario sampling logic in `work/datagen_wyp.py` into a deterministic scenario sampler with split metadata. The initial implementation is `src/rages_sampling.py`.
3. Done: implement deterministic IR sampler for training. The implementation is in `src/rages_sampling.py` and samples structured `IR = (sigma, dz, g, filters)` without using raw text or an LLM.
4. Implement hard filter and token/action mask builder using the existing graph-validity utilities as the first backend.
5. Done: promote `generate_traj_with_wyp` and `compute_metrics` from `work/datagen_reasoning.py` into library modules. `generate_traj_with_wyp` now lives in `src/optimization/optimization.py`; `verify_waypoint_plan` in `src/rages_sampling.py` provides deterministic logging; metric/scoring helpers live in `src/rages_scoring.py`.
6. Generate initial SCP rollout dataset using the current `work/datagen_wyp.py` workflow.
7. Train the current GMM/VAE `p_phi` baseline using reward-weighted MLE.
8. Train the RAGES+ flow-matching `p_phi` candidate and compare it against the GMM baseline.
9. Freeze the selected `p_phi` and define deterministic decoding.
10. Done (v0 code path): generate Stage 2 candidate dataset using explicit frozen `p_phi + SCP` via `work/datagen_q.py`; production run remains gated on the selected frozen Item 9 checkpoint.
11. Done (v0): train censoring-aware `Q_psi(s, a)` via `src/rages_q.py` + `work/train_q.py`.
12. Done (v0): build Q calibration/ranking fixture via `work/analysis_q.py`.
13. Done (v0): implement `rho_sigma(Q)` external scoring. The tolerance-based lexicographic ranking and pass-all hard filter live in `src/rages_scoring.py` (cf. Sec. 6.3), now with Q-output adapters for `p_conv` thresholding and metric means.
14. Train Stage 3a structured SFT policy.
15. Implement masked group sampling.
16. Implement Stage 3b GRPO using Q-based group rewards.
17. Implement periodic true-SCP audit.
18. Implement drift-loop relabeling and Q refit.
19. Run main experiments and ablations.

## 12.1 Repository workflow and validation pipeline

Use the current repository as the v1 baseline and add RAGES+ stages incrementally.

### Action schema and deterministic sampler validation

Purpose: verify that the RAGES+ action schema and deterministic sampler are usable without loading neural models or running SCP.

Expected workflow:

```text
PYTHONPATH=src python3 - <<'PY'
from rages_sampling import SplitConfig, sample_curated_rollout, sample_curated_rollouts

sample = sample_curated_rollout(0, seed=7, split="train")
print(sample.action.to_dict())
print(sample.curation.to_dict())
print([x.scenario.split for x in sample_curated_rollouts(
    5,
    seed=7,
    split_config=SplitConfig(train=0.6, val=0.2, test=0.2),
)])
PY
```

Required checks:

- `sample.action` contains only `b_seq` and `tof_steps`,
- `sample.curation` contains `policy`, `dt_orbits`, `dt_ranges`, and `target_domains`,
- repeated calls with the same `sample_id` and `seed` produce the same result,
- split labels are assigned from `SplitConfig`.

### IR sampler validation

Purpose: verify that training-time IRs are sampled as structured data without calling an LLM or consuming raw text.

Expected workflow:

```text
PYTHONPATH=src python3 - <<'PY'
from rages_sampling import sample_curated_rollout, sample_ir, sample_ir_batch

rollout = sample_curated_rollout(0, seed=7, split="train")
ir = sample_ir(0, seed=11, scenario=rollout.scenario, split="train", profile="auto")
print(ir.to_dict())
print([x.profile for x in sample_ir_batch(5, seed=11)])
PY
```

Required checks:

- `ir.ir.sigma.priority` is a permutation of `fuel`, `time`, `observation`, and `safety_margin`,
- `ir.ir.dz` contains terminal-domain or direction intent,
- `ir.ir.g.task_class` is one of the supported task classes,
- `ir.ir.filters` contains only structured hard-filter fields,
- repeated calls with the same `sample_id`, `seed`, and `profile` produce the same result.

### V1 smoke and artifact validation

Purpose: verify that the checked-in model/data paths and imports are usable.

Expected workflow:

```text
python src/rages.py --idx 10
```

Required checks:

- `src/rages.py` loads `model/reasoning_model/...`, `model/wyp_model/...`, and `data/wyp_data/...`,
- waypoint inference returns `x_pred` and `dt_pred`,
- SCP returns `status_cvx` and `status_scp`,
- output JSON is serializable.

### Stage 1 waypoint baseline validation

Current baseline workflow:

```text
python work/datagen_wyp.py
python work/train_wyp_predictor.py
python work/analysis_wyp_vs_random.py
```

Required metrics:

- SCP convergence rate,
- fuel / delta-v,
- observation score,
- safety margin,
- waypoint-domain error,
- deterministic decode repeatability.

The GMM/VAE baseline should remain pinned as a reproducibility fixture before flow matching is introduced.

### Stage 1 flow-matching validation

New flow-matching workflow should mirror the existing waypoint scripts:

```text
work/train_wyp_flow.py
work/analysis_wyp_flow.py
```

Required metrics:

- deterministic ODE decode SCP convergence,
- multi-sample SCP convergence distribution,
- best-of-K and average-of-K utility under true SCP,
- diversity of waypoint modes for fixed `(s, a)`,
- invalid-simplex or invalid-domain rate,
- comparison against `model/wyp_model/model_gmm_v3_weighted_one_hot.pt` and `model/wyp_model/model_gmm_v4_weighted_one_hot.pt`.

Do not use mean waypoint prediction as the main flow-matching evaluation. Mean prediction is only a baseline diagnostic.

### Stage 2 Q validation

New Q workflow should consume all retained candidates from frozen `p_phi + SCP` rollouts:

```text
work/datagen_q.py
work/train_q.py
work/analysis_q.py
```

Required metrics:

- convergence ECE,
- Brier score or NLL,
- false-positive rate for predicted-feasible but SCP-failing candidates,
- metric RMSE/MAE on converged samples,
- within-scenario Kendall tau,
- top-1 regret against true-SCP ranking.

### Stage 3 policy validation

New structured-policy workflow:

```text
work/train_policy_sft.py
work/train_policy_grpo.py
work/analysis_rages_plus.py
```

Required metrics:

- mask-valid action rate,
- behavior graph feasibility,
- Q-ranked candidate utility under `rho_sigma`,
- true-SCP convergence of selected candidates,
- top-1 regret against oracle true-SCP ranking,
- number of true SCP calls per selected trajectory,
- policy latency.

### Parser validation

Parser work should stay separate from Stage 1--4:

```text
work/parse_intent_to_ir.py
work/analysis_parser.py
```

Required metrics:

- slot-level F1 for `sigma`, `dz`, `g`, and filters,
- command-class accuracy,
- abstain precision/recall,
- malformed-command rejection rate.

## 13. Main experiments

The main evaluation should compare:

1. full RAGES+ pipeline,
2. no-abstraction baseline,
3. SFT-only policy without Q-guided improvement,
4. argmax-only distillation baseline,
5. no-drift-loop baseline,
6. scalarization baseline,
7. oracle true-SCP ranking baseline,
8. Q without convergence calibration,
9. GMM waypoint baseline versus flow-matching waypoint model,
10. stochastic versus deterministic `p_phi` decoding.

Key metrics:

- SCP convergence rate,
- final utility under `rho_sigma`,
- individual physical metrics `m_j`,
- fuel / delta-v,
- time of flight,
- terminal error,
- safety margin,
- passive-safety or KOZ violations,
- number of SCP calls,
- inference latency,
- top-1 regret against oracle ranking,
- monitor-triggered replanning success rate.

## 14. Naming and positioning

Recommended terminology:

```text
SCP-grounded guidance affordance
optimization-grounded affordance
verifier-proxy-guided behavior proposal
intent-conditioned behavior abstraction
```

Avoid saying that Q certifies safety. Q only predicts verifier outcomes.

A concise method description:

```text
RAGES+ learns a behavior-level guidance policy by bootstrapping from SCP-certified trajectories. A frozen waypoint model maps behavior sequences to optimizer-facing constraints, while a sigma-free verifier surrogate predicts convergence and mission-quality metrics for each candidate behavior. Intent-dependent preferences are applied only through an external scoring function, and final feasibility remains certified by SCP with periodic audit and relabeling to prevent proxy drift.
```

## 15. Non-negotiable invariants

Keep these invariants throughout implementation:

```text
1. Q is sigma-free unless sigma changes the verifier outcome.
2. Q outputs preference-free metrics, not scalar utility.
3. rho_sigma performs preference-dependent scalarization outside Q.
4. SCP is the final feasibility authority.
5. The policy proposes structured behavior actions, not raw trajectories.
6. The parser is separate and never enters the learned guidance certification path.
7. All candidates are retained for Q training, including losers and failed samples.
8. Metric heads are trained only on converged samples unless failure metrics are explicitly defined.
9. p_phi decoding must be deterministic for Q-label generation, unless Q is explicitly defined as an expectation over stochastic decoding.
10. Lower-layer changes invalidate or version all dependent upper-layer artifacts.
```


## 16. Implementation status

Use this section to track completion of the agenda in Sec. 12.

Current validation status:

- Items 1--3 validated with lightweight deterministic sampler smoke tests.
- Item 5 validated with import/compile checks and a lightweight wrapper/scoring extraction smoke test. Full SCP execution is intentionally left to the Stage 2 rollout pipeline because it invokes the heavy solver stack.
- Item 6 pipeline is ready and validated with a 6-sample smoke test: all four metrics stored correctly, non-converged rows have NaN metrics and zero reward weight, determinism confirmed with seed re-run. The full production datagen run has not been executed yet.
- Parser workstream v0 validated with import/compile checks, heuristic single-parse smoke, and offline benchmark (`work/analysis_parser.py --backend heuristic`). Live OpenAI calls remain pending an installed OpenAI runtime/API key.
- Items 10--12 Stage 2 Q stack v0 validated with compile checks, 2-scenario x 2-candidate Q datagen smoke using an explicitly supplied v4 waypoint checkpoint, 2-epoch Q training smoke, analysis fixture output, and direct `rho_sigma` Q adapter smoke. The v4 checkpoint was smoke-only; production Stage 2 labels still require the frozen Item 9 `p_phi` checkpoint.
- Next structural target is running item 6 at full scale (`work/datagen_wyp.py`, N=80k), completing the item 7--9 freeze path, then running production `work/datagen_q.py` against that frozen checkpoint.

| Agenda item | Status | Implementation / notes |
| --- | --- | --- |
| 1. Behavior vocabulary and action schema | Done | Added `Action`, `ActionCurationMetadata`, and `CuratedAction` in `src/parameters.py`. `Action` is strictly `(b_seq, tof_steps)`; `policy`, `dt_orbits`, transfer windows, and target-domain traces are curation metadata only. |
| 2. Deterministic scenario sampler with split metadata | Done | Added `src/rages_sampling.py` with `SplitConfig`, `ScenarioSample`, `ScenarioRolloutSample`, `sample_curated_rollout`, and `sample_curated_rollouts`. |
| 3. IR sampler | Done | Added IR dataclasses and deterministic training sampler in `src/rages_sampling.py`: `IRSigma`, `IRDeltaZ`, `IRGoal`, `IRFilters`, `IR`, `IRSample`, `sample_ir`, `sample_ir_batch`, and `enumerate_priority_profiles`. This sampler is structured and LLM-free; raw-text parsing remains a separate parser workstream. |
| 4. Hard filter and token/action mask builder | Partial | IR-derived hard filter is the v0 pass-all placeholder `hard_filter_pass_all` in `src/rages_scoring.py` (cf. Sec. 3.1); grammar/admissibility stays with the behavior-graph sampler. Token/action mask builder not started. |
| 5. SCP verifier wrapper with deterministic logging | Done (thin wrapper) | Moved `generate_traj_with_wyp` into `src/optimization/optimization.py` as the library SCP execution helper. Added `WaypointPlan`, `SCPVerifierConfig`, `SCPVerifierResult`, and `verify_waypoint_plan` in `src/rages_sampling.py`; the wrapper imports the optimization helper directly and no longer needs `_load_scp_verifier_backend`. Migrated `compute_obs_score` and `compute_metrics` into `src/rages_scoring.py`, alongside `verifier_metric_row`, `verifier_feasible`, and `verifier_scoring_inputs`, so verifier outputs can feed `rho_sigma`. |
| 6. Initial SCP rollout dataset | Pipeline ready; full run pending | Rewrote `work/datagen_wyp.py` to use `sample_curated_rollout` + `verify_waypoint_plan` as the generation path. All rows (converged and failed) are retained: non-converged rows get `reward=0` and NaN metrics so they have zero weight in wyp training but are available for Q convergence-head training (cf. Sec. 5.4). Full 4-metric vector (`fuel_dv`, `transfer_time_sec`, `observation_score`, `safety_margin_m`) stored in a `(N, 4)` metrics tensor using the CT trajectory for `safety_margin_m`. Reward shift applied only to converged rows. Dataset saves to `data/wyp_data/data_v5.pth` with `seed` and `metric_keys` in meta. Smoke-tested at N=6: ~83% convergence, determinism confirmed, NaN/converged consistency verified. The production N=80k dataset has not been generated yet. |
| 6--8. Waypoint model library prep | Done (pure move) | Created `src/wyp_predictor.py` by moving code verbatim: `FillerConfig`, featurization (`build_input_from_data`, `build_input_slices`, ...), scaling/stats helpers, `ConditionalGMM`, `ConditionalVAE`, `masked_mdn_nll`, `constrained_fill`, `WypDataset` from `work/train_wyp_predictor.py`, and `load_model`, `build_data_from_values`, `predict_wyp_seq` from `work/datagen_reasoning.py`. The training script keeps only its training loop; importers (`src/rages.py`, `work/datagen_reasoning.py`, `work/analysis_rages.py`, `work/analysis_wyp_vs_random.py`) redirected. No functional changes; the flow-matching model class will be added to this module following the same `forward` / `compute_loss` / `sample_y` protocol. Validated: import smoke of all touched modules, v4 checkpoint load + `predict_wyp_seq`, and `python src/rages.py --idx 10` end-to-end. |
| 7--9. `p_phi` training and freeze path | Partial; script cleanup pending | `src/wyp_predictor.py` now supports `ConditionalGMM`, `ConditionalVAE`, and `ConditionalFlowMatcher` through the same `forward` / `compute_loss` / `sample_y` protocol, and `load_model()` already restores checkpoints by `model_type`. The remaining construction should be a small refactor of `work/train_wyp_predictor.py`, not a new training stack: expose `--model-type {gmm,vae,flow}`, paths, weighting, `K`, latent dimension, KL beta, and flow steps; add flow to the existing construction branch; guard debug logging that assumes `mu/std`; then train GMM/VAE/flow on `data_v5`, compare true-SCP metrics, and freeze the selected deterministic decode contract (`use_mean_w=True`). |
| 10. Parser workstream (Sec. 10) | Done (v0) | Added standalone text-to-IR parser in `src/rages_parser.py`, eval renderer/gold examples in `work/parser_eval_data.py`, CLI runner `work/parse_intent_to_ir.py`, and benchmark `work/analysis_parser.py`. The parser emits the same `IR` dataclasses as the sampler, completes partial priority orders with the schema-driven default tail, leaves `dz` unconstrained when no terminal intent is stated, and keeps raw text outside Stage 1--4 training. |
| 10--12. Stage 2 Q stack | Done (v0 code path); production labels pending frozen `p_phi` | Added grouped scenario/action candidate generation (`sample_scenario`, `enumerate_policy_paths`, `sample_candidate_actions`), `work/datagen_q.py`, `src/rages_q.py`, `work/train_q.py`, and `work/analysis_q.py`. Q input is only `(s, a)`, Q output is `{p_conv, metric_means}`, convergence BCE trains on all rows, and metric MSE trains only on converged finite metrics. Validated with compile checks, tiny SCP-backed datagen smoke, 2-epoch train smoke, analysis fixture, and Q-scoring adapter smoke. |
| 13. External scoring `rho_sigma(Q)` | Done (v0) | Added `src/rages_scoring.py`: tolerance-based lexicographic ranking (`LexicographicPreference`, `lexicographic_key`, `rank_candidates`, `select_best`) with epsilon-bucket quantization for transitivity, plus `epsilons_from_metric_spread` for tolerance calibration (cf. Sec. 6.3). `LexicographicPreference` now carries `p_conv_threshold`, and Q adapters (`q_metric_row`, `q_feasible`, `rank_q_candidates`, `select_best_q`) wire `Q_psi` outputs into the ranking path. Default epsilons are placeholders pending calibration on production Stage 2 data. |

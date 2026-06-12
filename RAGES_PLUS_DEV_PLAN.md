# RAGES+ Development Plan

This document summarizes the development procedure for the bootstrapped, bottom-up RAGES+ training pipeline. The goal is to implement the pipeline in a way that cleanly separates physical/optimization feasibility, intent-dependent preference, and parser-side language handling.

IMPORTANT: this is a research codebase for prototyping, so simplicity / readability of the codebase is critical, rather than a lot of assertions. Prioritize the simplicity of the overall code organization.

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
- waypoint model baseline training: `work/train_wyp_predictor.py`,
- frozen-waypoint candidate rollout and SCP metric computation: `work/datagen_reasoning.py`,
- reasoning SFT baseline: `work/train_reasoning_model.py`,
- end-to-end smoke wrapper: `src/rages.py`.

RAGES+ should reuse these components, but it should introduce explicit library-level interfaces for `Scenario`, `IR`, `Action`, `WaypointPlan`, `VerifierResult`, `QOutput`, and artifact versions. The current scripts can remain as v1 baselines, while new RAGES+ modules should avoid depending on ad hoc JSON rows or script-local globals.

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

The current repository baseline is the conditional GMM/VAE waypoint filler in:

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

1. Promote the existing behavior vocabulary and campaign graph in `src/parameters.py` into an explicit action schema.
2. Promote the existing scenario sampling logic in `work/datagen_wyp.py` into a deterministic scenario sampler with split metadata.
3. Implement IR sampler.
4. Implement hard filter and token/action mask builder using the existing graph-validity utilities as the first backend.
5. Promote `generate_traj_with_wyp` and `compute_metrics` from `work/datagen_reasoning.py` into an SCP verifier wrapper with deterministic logging.
6. Generate initial SCP rollout dataset using the current `work/datagen_wyp.py` workflow.
7. Train the current GMM/VAE `p_phi` baseline using reward-weighted MLE.
8. Train the RAGES+ flow-matching `p_phi` candidate and compare it against the GMM baseline.
9. Freeze the selected `p_phi` and define deterministic decoding.
10. Generate Stage 2 candidate dataset using frozen `p_phi + SCP`.
11. Train censoring-aware `Q_psi(s, a)`.
12. Build Q calibration/ranking fixture.
13. Implement `rho_sigma(Q)` external scoring.
14. Train Stage 3a structured SFT policy.
15. Implement masked group sampling.
16. Implement Stage 3b GRPO using Q-based group rewards.
17. Implement periodic true-SCP audit.
18. Implement drift-loop relabeling and Q refit.
19. Run main experiments and ablations.

## 12.1 Repository workflow and validation pipeline

Use the current repository as the v1 baseline and add RAGES+ stages incrementally.

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


## 16. Note 

Record the edits in this section when you complete the agenda in Sec. 12 to track the high-level change. 
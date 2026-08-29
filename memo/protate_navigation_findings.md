# pRotatE Continuous Navigation: Findings, Current Working Approach, and Open Questions

**Branch:** `temporary/protate-navigation-patches`  
**Repository:** `HernandezEduin/MultiHopKG`  
**Status:** Experimental / debugging branch; `main` remains unchanged.  
**Last updated:** 2026-08-30

---

## 1. Purpose of this memo

This document summarizes the work required to make supervised continuous navigation work with **pRotatE** in MultiHopKG.

It is written for a reader who understands the implementation on `main`, especially the TransE-oriented continuous navigation pipeline, but has not followed the experimental branch.

The current conclusion is that the major obstacle was **not pRotatE itself**, but a mismatch between the geometry used by the pretrained pRotatE scoring function and the coordinate system presented to the navigation policy.

The current best approach is to perform policy navigation in a **canonical pi-periodic quotient space** while preserving the original pretrained pRotatE KGE and its retrieval semantics.

---

## 2. Baseline problem

The original continuous navigation code was developed primarily around TransE-like Euclidean geometry.

For TransE, the intended transition is approximately:

```text
head + relation -> tail
```

and ordinary linear differences are meaningful supervision targets.

For pRotatE, however, the pretrained score is based on phase coordinates and has the form

```text
abs(sin(h + r - t))
```

up to the model's modulus and margin terms.

Therefore pRotatE does **not** behave like an ordinary Euclidean embedding space.

Most importantly:

```text
theta and theta + pi
```

are equivalent under the scoring function because

```text
abs(sin(theta + pi)) == abs(sin(theta)).
```

This pi-periodicity is the central geometric fact behind the working solution.

---

## 3. Problems found in the original pRotatE navigation path

### 3.1 Action-coordinate / scaling mismatch

The navigation policy emits bounded actions through `tanh`, so actions live in approximately:

```text
[-1, 1]
```

The old pRotatE navigation path mixed:

- raw stored pRotatE embedding coordinates,
- phase values in radians,
- policy-normalized actions.

This could lead to phase displacements being effectively rescaled twice.

The corrected navigation convention is:

```text
action = phase_displacement / pi
```

so:

```text
action = +1 -> +pi radians
action = -1 -> -pi radians
```

The corrected transition is conceptually:

```python
head_rad = denormalize_embedding(state)
rotation_rad = action * pi
tail_rad = wrap(head_rad + rotation_rad)
tail = normalize_embedding(tail_rad)
```

The action itself must **not** be passed through `denormalize_embedding` again.

---

### 3.2 Double-tanh policy bug

The original `ContinuousPolicyGradient` behavior was effectively:

```python
mu = tanh(mu_raw)
dist = Normal(mu, sigma)
z = dist.rsample()
action = tanh(z)
```

Supervised training compared `mu` directly against the target action, but deterministic execution produced roughly `tanh(mu)`.

This implies a target of `1.0` can only execute deterministically as approximately:

```text
tanh(1) = 0.761594
```

The temporary pRotatE policy patch changes the logic so that the Gaussian is parameterized in unconstrained space and the output is squashed once:

```python
mu_raw = mu_layer(...)
mu = tanh(mu_raw)
dist = Normal(mu_raw, sigma)
z = dist.rsample()
action = tanh(z)
```

This bug was real and severe, but fixing it alone did **not** solve pRotatE navigation.

---

## 4. pRotatE-compatible retrieval / navigation geometry

The original generic nearest-neighbor / intrinsic-distance logic was not fully aligned with pRotatE.

The temporary navigation patch uses the same periodic structure as the pRotatE score:

```text
distance(target, entity) = sum(abs(sin(target_phase - entity_phase)))
```

This is used for pRotatE ANN lookup and intrinsic navigation distance.

The ANN patch also fixes rollout-shaped inputs and preserves the expected three-return-value interface:

```text
(resulting_embeddings, indices, distances)
```

Real-data diagnostics showed the corrected transition and ANN geometry are internally consistent.

---

## 5. Oracle transition diagnostic

Using exact endpoint-derived pRotatE displacements, the corrected transition reconstructs the target phase essentially exactly.

Observed on Kinship:

```text
oracle isolated-hop Hits@1 = 1.0
oracle round-trip max phase error ~= 4.8e-7
```

Therefore the corrected continuous transition itself is not the bottleneck.

---

## 6. Why endpoint-derived supervised actions are problematic

The `main`-style supervised objective constructs an action from each particular head/tail pair:

```python
target_action = difference(current_entity, gold_tail)
```

This is reasonable in TransE.

For pRotatE it creates a structural problem: different linear representatives can be equivalent under the pRotatE score.

Diagnostics comparing endpoint-derived targets against pretrained pRotatE relation embeddings found:

```text
endpoint target vs relation linear RMSE                ~= 0.7825
endpoint target vs relation mean abs(sin) phase error ~= 0.1319
fraction linearly far but periodically close           ~= 30.3%
within-relation mean component std                     ~= 0.5211
```

Thus ordinary MSE was penalizing differences that pRotatE itself often regards as equivalent.

---

## 7. Relation embeddings as semantic action targets

The next approach was to supervise each hop using the pretrained pRotatE relation phase itself instead of deriving a separate action from each head/tail pair.

Conceptually:

```text
A --r1--> B --r2--> C

state(A) -> predict pretrained r1
state(B) -> predict pretrained r2
```

The initial relation-action representation was:

```text
action = wrapped_relation_phase / pi
```

with values in approximately `[-1, 1]`.

This removes instance-specific endpoint ambiguity and gives every occurrence of the same relation the same semantic target.

---

## 8. Pretrained pRotatE relation actions are strong

A key diagnostic tested the pretrained relation rotations directly on Kinship.

### Isolated hops

```text
relation Hits@1 = 0.925403
relation Hits@3 = 1.000000
relation MRR    = 0.962702
```

### Full 2-hop relation composition

```text
relation-path Hits@1 = 0.729839
relation-path Hits@3 = 0.987903
relation-path Hits@10 = 1.000000
relation-path MRR = 0.855175
```

This is critical evidence:

> The pretrained pRotatE space itself is quite suitable for the navigation task.

The poor learned navigation results were therefore caused primarily by the policy representation / supervision pipeline rather than by an unusable KGE.

---

## 9. Early relation-target training results

### Relation targets + gold intermediate entity teacher forcing

At 50 epochs:

```text
Hits@1 ~= 11.29%
```

At 500 epochs with the inherited adapter auxiliary objective enabled:

```text
Hits@1 = 20.97%  (13 / 62)
```

At 2500 epochs:

```text
Hits@1 = 25.81%  (16 / 62)
Best epoch = 2382
```

So relation-target training was directionally correct but learned very slowly and remained far below the pretrained relation-composition capability.

---

## 10. Adapter auxiliary loss finding

The base residual-adapter auxiliary target is approximately:

```text
concat(head_embedding, mean(relation_embeddings_across_hops))
```

This is questionable for pRotatE because:

1. raw phase embeddings are cyclic;
2. ordinary arithmetic averaging is not phase-aware;
3. relation order is lost;
4. the objective was inherited from the TransE-oriented setup.

Setting:

```yaml
supervised_adapter_scalar: 0.0
```

removes the explicit adapter MSE objective.

Important: this does **not** freeze the adapter. The adapter still receives gradients through the main policy loss because `q_projected` is part of the policy state and its parameters remain in the optimizer.

At 500 epochs, adapter-off improved the result to:

```text
Hits@1 = 24.19%  (15 / 62)
Best epoch = 205
```

This nearly matched the 2500-epoch adapter-on result.

Current pRotatE quotient experiments therefore use:

```yaml
supervised_adapter_scalar: 0.0
```

unless explicitly testing the auxiliary objective.

---

## 11. Train/inference state-distribution mismatch

A checkpoint diagnostic compared hop-2 relation prediction in two regimes:

- **gold state:** exact annotated intermediate entity is provided;
- **free state:** the state produced by the policy's first predicted continuous action is used.

For the 2500-epoch adapter-on relation-target model:

```text
Hop 1 relation Hits@1               = 51.61%
Hop 2 relation Hits@1, gold state   = 54.84%
Hop 2 relation Hits@1, free state   = 25.81%
```

For the 500-epoch adapter-off model:

```text
Hop 1 relation Hits@1               = 38.71%
Hop 2 relation Hits@1, gold state   = 45.16%
Hop 2 relation Hits@1, free state   = 19.35%
```

This showed a severe exposure / state-distribution mismatch.

Training had been:

```text
A
 -> predict r1
 -> replace state with exact stored B
 -> predict r2
```

while inference is:

```text
A
 -> predicted continuous action
 -> continuous position B'
 -> predict r2
```

---

## 12. Continuous gold-relation state experiment

To reduce this mismatch, a variant advanced the training state by applying the **gold pretrained relation action continuously** instead of snapping to the exact annotated entity.

Conceptually:

```text
A
 -> supervise r1
 -> apply GOLD r1 continuously
 -> B'
 -> supervise r2 from B'
```

This avoids compounding the policy's own prediction error during training while exposing hop 2 to continuous states.

Result at 500 epochs, adapter-off:

```text
Hits@1 = 20.97%
Best epoch = 385
```

The experiment did not improve final accuracy.

However, its diagnostic showed that the state distribution had actually changed in the intended direction:

```text
Hop 2 relation Hits@1, exact gold entity state = 20.97%
Hop 2 relation Hits@1, free predicted state     = 27.42%
```

The previous gold-vs-free gap reversed.

Unfortunately hop-1 relation accuracy also fell, suggesting that continuous states were exposing the shared policy to a harder coordinate system.

This led to the central geometric insight below.

---

## 13. Central finding: the policy must respect pRotatE's pi quotient geometry

pRotatE regards phases differing by pi as equivalent:

```text
theta == theta + pi   (under the pRotatE score)
```

but the policy on the earlier branch variants consumed raw phase representatives directly.

Therefore two pRotatE-equivalent states such as:

```text
+0.9*pi
-0.1*pi
```

could look numerically very different to the policy even though the KGE treats them as equivalent.

The same ambiguity applied to relation-action targets.

This means the policy was learning over a non-unique coordinate system with artificial discontinuities.

---

## 14. Canonical pi-quotient representation

A canonical representative for the pRotatE equivalence class is defined as:

```text
theta_c = 0.5 * atan2(sin(2*theta), cos(2*theta))
```

which maps phase values into:

```text
[-pi/2, pi/2]
```

The normalized policy action becomes:

```text
a_c = theta_c / pi
```

so canonical actions are guaranteed to lie in:

```text
[-0.5, 0.5]
```

Example mappings:

```text
+0.90 -> -0.10
-0.82 -> +0.18
+0.43 -> +0.43
```

These are different numerical representatives of the same pRotatE relation classes.

Entity states are canonicalized in the same pi-periodic quotient geometry before they are exposed to the policy.

---

## 15. Quotient diagnostic: canonicalization is lossless for pRotatE retrieval

A real-data diagnostic compared standard relation rotations against canonical pi-quotient rotations.

### Action magnitude

```text
Standard mean |component|  = 0.528834
Canonical mean |component| = 0.155364

Standard max |component|   = 0.995361
Canonical max |component|  = 0.496035

Standard fraction |a| > 0.5 = 0.557796
Canonical fraction |a| > 0.5 = 0.0
```

Thus the canonical target is substantially easier numerically.

### Isolated-hop retrieval

```text
Standard Hits@1  = 0.925403
Canonical Hits@1 = 0.925403

Standard Hits@3  = 1.000000
Canonical Hits@3 = 1.000000

Standard MRR     = 0.962702
Canonical MRR    = 0.962702
```

### 2-hop relation composition

```text
Standard Hits@1  = 0.729839
Canonical Hits@1 = 0.729839

Standard Hits@3  = 0.987903
Canonical Hits@3 = 0.987903

Standard MRR     = 0.855175
Canonical MRR    = 0.855175
```

### State equivalence

After canonicalization, standard and quotient transitions agree numerically to approximately:

```text
max absolute disagreement       ~= 1.34e-7
mean max absolute disagreement  ~= 3.99e-8
```

Therefore quotient canonicalization does **not** reduce the underlying pretrained pRotatE capability.

It simply chooses a unique and easier coordinate system for the policy.

---

## 16. Current working approach

The current successful method is **pi-quotient pRotatE navigation**.

### Supervised action target

For each annotated relation:

```text
raw pretrained pRotatE relation
 -> convert to phase
 -> canonicalize modulo pi
 -> divide by pi
 -> policy target in [-0.5, 0.5]
```

### Policy state

Continuous pRotatE entity positions are canonicalized modulo pi before being exposed to the policy.

### Multi-hop warm-up

Current conceptual training flow:

```text
Natural-language question
       |
       v
BERT / question encoder
       |
       v
q_projected / residual adapter
       |
       +-------------------------+
       |                         |
       v                         |
canonical starting pRotatE state|
       |                         |
       v                         |
policy predicts canonical r1    |
       |                         |
       +-- supervised against canonical pretrained r1
       |
       v
apply GOLD canonical r1 continuously
       |
       v
canonicalize resulting state
       |
       v
policy predicts canonical r2
       |
       +-- supervised against canonical pretrained r2
```

The underlying KGE itself is not retrained or altered.

ANN / entity retrieval remains based on the pRotatE-compatible periodic distance:

```text
sum(abs(sin(target_phase - entity_phase)))
```

Thus quotient canonicalization changes **what the policy sees and predicts**, not what the pretrained pRotatE model considers semantically equivalent.

---

## 17. Current best supervised-navigation result

Run command:

```bash
python nav_supervised_training_protate_quotient.py \
    --preferred_config=./configs/supervised_path_learning/KinshipHinton_protate_test.yaml
```

Current recommended debugging settings:

```yaml
epochs: 500
supervised_adapter_scalar: 0.0
```

Best checkpoint from the first quotient experiment:

```text
models/nav_sv/20260830_014955
```

Best validation epoch:

```text
253
```

Metrics:

```text
Hits@1  = 0.61290
Hits@3  = 0.61290
Hits@5  = 0.61290
Hits@10 = 0.61290
Hits@20 = 0.61290
MR      = 39.70968
MRR     = 0.61674
distance = 0.86709
```

With 62 evaluation examples:

```text
38 / 62 are solved at rank 1.
```

The metric pattern is exactly consistent with 38 rank-1 successes and 24 complete misses under the evaluator's `num_rollouts_test + 1 = 101` missing-answer sentinel:

```text
(38*1 + 24*101) / 62 = 39.70968
```

---

## 18. Improvement relative to previous pRotatE formulations

At the same 500-epoch debugging budget:

| Formulation | Hits@1 |
|---|---:|
| Earlier endpoint-target pRotatE | roughly 6-8% |
| Relation targets + gold entity TF + adapter ON | 20.97% |
| Relation targets + gold entity TF + adapter OFF | 24.19% |
| Relation targets + continuous gold-relation state | 20.97% |
| **Pi-quotient states + actions** | **61.29%** |

The quotient formulation improves over the strongest previous 500-epoch run by:

```text
61.29 - 24.19 = 37.10 percentage points
```

or approximately:

```text
61.29 / 24.19 ~= 2.53x
```

This is currently the strongest evidence that the principal pRotatE navigation problem was a **representation-geometry mismatch**.

---

## 19. Important distinction: relation-composition reference is not yet an exact test-set ceiling

The direct pretrained relation-composition diagnostic gave:

```text
2-hop Hits@1 ~= 72.98%
```

but that number was computed over the full set of 248 2-hop QA examples.

The supervised navigation result above is evaluated on 62 examples.

Therefore 72.98% should currently be called a **reference / pretrained relation-composition capability**, not the exact upper bound for the 62-example navigation test split.

A next diagnostic should compute direct quotient relation composition on the exact same 62 examples used by the navigation evaluator.

---

## 20. Important data-split caveat

For the small Kinship split, the existing data-loading logic contains:

```python
if len(test_df) < 100:
    dev_df = test_df
```

Since the relevant test set contains 62 examples:

```text
dev_df == test_df
```

This explains why validation and test metrics are identical.

It also means checkpoint selection uses the same 62 examples later reported as test.

Therefore:

> The 61.29% result is valid as a controlled debugging / ablation comparison on this branch, but must not yet be presented as an unbiased final test result.

Before paper-quality reporting, use a truly independent validation set or a fixed train/dev/test split.

---

## 21. Current implementation pieces on the temporary branch

Important experimental files include:

```text
temporary_patches/protate_navigation.py
    pRotatE-compatible transition, difference, periodic distance, ANN search

temporary_patches/protate_policy.py
    single-tanh / unconstrained Gaussian-mean policy compatibility patch

temporary_patches/protate_supervision.py
    relation-target supervision and continuous-state variants

nav_supervised_training_protate.py
    relation-target + gold-entity teacher-forcing experiment

nav_supervised_training_protate_continuous.py
    relation-target + continuous gold-relation state experiment

nav_supervised_training_protate_quotient.py
    current quotient-space training entry point

diagnostics/protate_navigation_realdata.py
    real-data transition / relation diagnostics

diagnostics/protate_policy_checkpoint.py
    per-hop trained policy diagnostic for earlier non-quotient variants

diagnostics/protate_quotient_realdata.py
    validates lossless pi-quotient relation/state canonicalization

tests/test_protate_navigation_patch.py
    focused unit tests for pRotatE navigation behavior
```

The temporary entry points intentionally avoid modifying the normal `main` training path while the approach is still being validated.

---

## 22. Important commits in this branch history

Notable commits in the experimental sequence include:

```text
a4ee40f   Normalize pRotatE phase actions and navigation geometry
ed64d66   Enable temporary pRotatE supervised-training compatibility
458136b   Add pRotatE normalized/cyclic transition tests
7fbc448   Align pRotatE absolute-difference geometry
bb3d34c   Add real-data pRotatE navigation diagnostic
3db4b96   Add multi-hop composition / ambiguity diagnostic
9215192   Add single-tanh pRotatE policy patch
c63876b   Install policy patch from pRotatE entry point
0c709ee   Relation-phase targets + gold-state teacher forcing
a39b628   Wire relation-target teacher forcing into pRotatE entry point
8ae94df   Add trained-policy per-hop diagnostic
21373df   Add continuous gold-relation state supervision
c9dcc65   Add continuous-state pRotatE training entry point
dc7c3cc   Validate pi-quotient canonical actions on real data
2d554c3   Add quotient-space pRotatE navigation geometry
e419b5a   Add quotient-space supervised-training entry point
```

This list is intended as a navigation aid rather than a guarantee that every intermediate commit remains independently desirable. The current quotient approach supersedes several earlier experimental formulations.

---

## 23. Warnings / cleanup items

Two non-fatal warnings were observed repeatedly:

### Tensor creation from a list of numpy arrays

Earlier variants emitted a warning from `temporary_patches/protate_supervision.py` when doing:

```python
paths_t = torch.tensor(paths, ...)
```

where `paths` could be a list of NumPy arrays.

Prefer converting once through `np.asarray(...)` before `torch.tensor(...)` if that code path remains.

### Tensor copy warning in `pn.py`

The environment currently has code similar to:

```python
entity_indices = torch.tensor(entity_indices, dtype=torch.int)
```

when `entity_indices` may already be a tensor.

Prefer `.to(dtype=...)`, `.clone()`, or `.detach().clone()` as appropriate.

These warnings are not believed to explain the pRotatE performance differences, but should be cleaned up before merging production code.

---

## 24. What has been ruled out / clarified

### pRotatE is not fundamentally incapable of navigation

Direct relation composition reaches approximately 73% Hits@1 on the full 2-hop Kinship QA set.

### The corrected transition is not the main failure

Oracle endpoint displacements reconstruct tails essentially exactly.

### Exact pi-separated duplicate entities are not driving the result

Earlier diagnostics found no exact entity aliases separated purely by the pi equivalence in the tested entity set.

### Double-tanh was real but not the sole bottleneck

Fixing it was necessary, but policy performance remained poor without geometric canonicalization.

### Endpoint-target MSE is structurally poor for pRotatE

Large linear differences often correspond to small periodic pRotatE errors.

### Gold-entity teacher forcing creates a measurable train/inference state mismatch

Hop-2 relation accuracy fell sharply when evaluated on continuously predicted states.

### Continuous-state exposure alone is insufficient

Without canonicalization, continuous states are represented by arbitrary pi-equivalent coordinates and can make learning harder.

### Quotient canonicalization preserves pretrained KGE behavior

It leaves isolated-hop and multi-hop relation-composition retrieval unchanged while dramatically reducing action magnitude and removing duplicate numerical representatives.

---

## 25. Recommended next experiments

### Priority 1: quotient-aware checkpoint diagnostic

The existing `diagnostics/protate_policy_checkpoint.py` was designed around the earlier non-quotient representation.

Create a quotient-aware version that reports, for each hop:

```text
canonical relation Hits@1 / Hits@3
canonical action linear error
canonical periodic error
intermediate gold-tail entity rank
relation prediction from gold-canonical state
relation prediction from free-canonical state
```

and final:

```text
deterministic answer Hits@1 / Hits@3 / MRR
sampled answer Hits@1
```

This should localize the remaining errors after the large quotient improvement.

### Priority 2: exact direct-relation reference on the same 62 examples

Compute canonical pretrained relation composition on exactly the same evaluation split used by the navigation checkpoint.

This gives the correct split-specific relation-composition reference rather than comparing 62 examples against the 248-example full-data diagnostic.

### Priority 3: independent dev/test split

Fix the current `dev_df = test_df` behavior for small datasets before reporting final numbers.

### Priority 4: rerun multiple seeds

The current breakthrough result should be replicated across several random seeds before drawing variance-sensitive conclusions.

### Priority 5: longer quotient training

The first quotient model already reached 61.29% with its best checkpoint at epoch 253 of 500.

A longer run is reasonable only after confirming that the evaluation protocol and quotient diagnostic are correct. Unlike the earlier non-quotient setup, it may no longer require thousands of epochs.

### Priority 6: sigma / rollout analysis

The supervised expected sigma is extremely small (`~1e-5` in previous diagnostics), so many test rollouts may be nearly duplicates.

Investigate:

```yaml
supervised_sigma_scalar: 0
```

or a deliberately calibrated nonzero exploration target if diverse trajectories are desired.

### Priority 7: reconsider the adapter auxiliary objective

If an explicit adapter objective is restored, it should be redesigned for periodic / ordered relation structure rather than using a linear mean of raw relation embeddings.

Possible future formulations include:

- phase-aware composed relation targets;
- canonical quotient relation composition;
- sequence-aware relation/path encoders;
- doubled-angle features such as `[cos(2*theta), sin(2*theta)]`.

---

## 26. Current practical recommendation

For further pRotatE supervised navigation experiments on this branch, start from:

```bash
python nav_supervised_training_protate_quotient.py \
    --preferred_config=./configs/supervised_path_learning/KinshipHinton_protate_test.yaml
```

with:

```yaml
epochs: 500
supervised_adapter_scalar: 0.0
```

and treat the quotient-space formulation as the current working baseline.

Do **not** return to raw phase representatives or ordinary endpoint-target MSE unless running an explicit ablation.

---

## 27. Current conceptual conclusion

The strongest current interpretation is:

> A continuous navigation policy over a periodic KGE should operate in a representation that respects the equivalence classes induced by the KGE scoring geometry.

For pRotatE, the raw phase coordinate is not a unique semantic state because values separated by pi are equivalent under `abs(sin(.))`.

Using arbitrary raw representatives forces the policy to learn artificial discontinuities and large, redundant action targets.

Canonicalizing both states and relation actions into the pRotatE pi-periodic quotient space:

```text
[-pi/2, pi/2]
```

or equivalently normalized policy coordinates:

```text
[-0.5, 0.5]
```

preserves the pretrained KGE's retrieval ability while making the policy-learning problem dramatically easier.

The first controlled experiment improved supervised navigation Hits@1 from the strongest previous 500-epoch result of **24.19%** to **61.29%** without retraining the underlying pRotatE KGE.

That is the current central finding of this branch.

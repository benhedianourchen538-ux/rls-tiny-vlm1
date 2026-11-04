# E3 — One pooled visual token vs an 8×8 grid of visual tokens

**Question.** Does collapsing the image to a single vector hurt relations (and which attributes) more than a grid?
**Hypothesis (REWRITE IN YOUR OWN WORDS, COMMIT BEFORE RUNNING).** Proposed: colour/shape/size survive pooling (global
average keeps "what"), but relations (leftof/above...) and which-object-has-which-colour need "where" and binding, so the
relation accuracy and 2-object exact match drop clearly while 1-object exact match drops little.
If I'm wrong (relations unaffected), the CNN's 128 channels + padding already encode position, or pooling kept it.
**Baseline.** `configs/baseline.yaml`. **One change.** `visual_tokens: global` (1 token instead of 64).
**Seeds.** 0 1 2 per setup. **Command.** see README. **Result.** mean ± std per attribute + exact_match_by_n_objects.
**Interpretation / limits.** …

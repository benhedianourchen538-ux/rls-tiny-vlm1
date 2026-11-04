# E1 — Blind baseline

**Question.** Does the model actually use the image?
**Hypothesis (REWRITE IN YOUR OWN WORDS, COMMIT BEFORE RUNNING).** Proposed: with zero images the model can only learn
the *grammar* of the word (size→color→shape→[relation→size→color→shape]) and priors. I expect: size/color/shape/relation
accuracy near chance (1/2, 1/4, 1/4, 1/4 — slightly higher for the relation slots only through length priors), exact match
~0 %, but teacher-forced letter accuracy still high (~60–70 %) because most letters inside a part follow from spelling.
If I'm wrong and attribute accuracy is far above chance, the "blind" model leaks information (check `image_mode`).
**Baseline.** `configs/baseline.yaml`. **One change.** `image_mode: zeros` (everything else identical). **Seeds.** 0 1 2.
**Command.** `python run_seeds.py --config configs/blind.yaml --name blind --seeds 0 1 2 --out experiments/E1_blind`
**Result.** table of per-attribute accuracy (blind vs baseline) + letter accuracy; **Interpretation / limits.** …

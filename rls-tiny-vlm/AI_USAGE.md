# AI usage

**Tool:** Claude (Anthropic), used as a coding assistant.

**What it did:** produced the first version of the repository skeleton and the code in `src/`, `benchmarks/`,
`experiments/` and the docs, from the Learning Guide and Project Brief.

**What I (the author) must do before submitting — tick each when true, delete what is not:**
- [ ] I read every file in `src/` and can redraw the tensor shapes from image to logits without looking.
- [ ] I re-derived the attention by hand and ran `pytest` myself; I can explain why a test would fail if I removed `/ sqrt(d_head)` or the mask.
- [ ] I changed at least one thing on purpose (e.g. mask, number of pools, learning rate) and predicted the effect first.
- [ ] I wrote the hypotheses in `experiments/*/notes.md` in my own words BEFORE running (commit timestamps).
- [ ] The numbers in README/report come from my own runs on hardware I name. Anything I did not run is not in the report.
- [ ] Debugging episodes (what broke, how I found it) are described in the notes.

Write here, honestly, what you rewrote or rejected: …

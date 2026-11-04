# RLS Entrance Challenge — Tiny Vision-Language Model

A 1.07 M-parameter model that looks at a 64×64 ShapeScenes image and spells a one-word description
(`smallbluesquareleftoflargeyellowtriangle`) letter by letter. CNN encoder → adapter → Transformer decoder with
**handwritten** multi-head attention. No pretrained weights, no `nn.MultiheadAttention`, no `nn.Transformer*`.

## Results

> Fill every `TBD` from the `*_results.json` / CSV files produced by the commands below. Do not type numbers by hand.

| Experiment | Configuration | Metric | Result (mean ± std, n seeds) | Interpretation (1 sentence) |
|---|---|---|---|---|
| Main model | `configs/baseline.yaml` | Exact match, test | TBD | TBD |
| E1 blind | `configs/blind.yaml` | Attribute acc., test | TBD | TBD |
| S1 throughput | batch 64, TBD GPU | images/s | TBD | TBD |

## Reproduce

```bash
pip install -r requirements.txt
python generate_data.py                      # fixed seed 42 -> data/{train,val,test,test_heldout}.pt (do not change)
pytest -q                                    # attention == F.scaled_dot_product_attention, shapes, causality, gradients

python -m experiments.E0_overfit.overfit --device cpu              # E0: overfit one batch -> overfit_loss.png
python run_seeds.py --config configs/baseline.yaml --name baseline --seeds 0 1 2 --out experiments/main
python run_seeds.py --config configs/blind.yaml    --name blind    --seeds 0 1 2 --out experiments/E1_blind
python run_seeds.py --config configs/e3_global.yaml --name e3_global --seeds 0 1 2 --out experiments/E3_custom

python -m benchmarks.S1_throughput.bench --device cpu               # then --device cuda
python -m benchmarks.S1_throughput.plot
python -m benchmarks.level2.profile_step --device cuda              # step breakdown + torch.profiler
python -m benchmarks.level2.memory --device cuda                    # estimate vs measured peak memory
python -m benchmarks.level2.attention_bench --device cuda           # ours vs F.scaled_dot_product_attention
```
Single run: `python -m src.train --config configs/baseline.yaml --seed 0 --out runs/b0 [--resume] [--device cpu|cuda]`.
Evaluate: `python -m src.evaluate --ckpt runs/b0/best.pt --split test_heldout`. A GPU/CPU switch is the single
`--device` flag. Colab: `notebooks/colab_run.ipynb`. Seeds are fixed through the config (`seed`) and `--seed`.

## Architecture (shapes for the default config, B = batch)

```
image uint8 (B,3,64,64) → /127.5-1 → 4×[Conv3x3-BN-ReLU] with 3 max-pools → feature map (B,128,8,8)
→ flatten + Linear(128→128) ("adapter") → visual tokens (B,64,128)
→ concat with letter embeddings (B,L,128)      → sequence (B,64+L,128)  (+ learned positional embedding)
→ 4 × pre-LN decoder block [masked MHA(4 heads) + MLP(4×) + residuals]  → (B,64+L,128)
→ Linear(128→27) applied from the LAST visual token onward → logits (B,L+1,27)
```
Mask ("prefix-LM"): visual tokens see all visual tokens, never letters; letter *i* sees all visual tokens and letters ≤ *i*.
Position 63 (last visual token) predicts letter 1; position 63+k predicts letter k+1; the last one predicts `<eos>`.
Loss: cross-entropy on those L+1 outputs, padded positions set to `ignore_index=-100`.

## Layout

```
generate_data.py        provided by RLS, unmodified
configs/                baseline.yaml, blind.yaml, e3_global.yaml
src/tokenizer.py        26 letters + <eos>, batching with padding
src/data.py             Dataset, Collator, DataLoader
src/model/encoder.py    CNN                 attention.py  handwritten MHA
src/model/decoder.py    block + stack + mask   vlm.py     encoder + adapter + decoder + head
src/train.py            loop, schedule, checkpoints/resume, curves
src/evaluate.py         exact match, per-attribute acc., letter acc.   parsing.py  robust word parser
src/generate.py         greedy decoding
tests/                  provided tests (attention equivalence, shapes, causality, generator)
experiments/            E0_overfit, E1_blind, E3_custom (config + notes.md + results.json + plots)
benchmarks/             S1_throughput, level2 (profile_step, memory, attention_bench)
docs/CONCEPTS.md        what to understand for the interview;  report/  report outline
```

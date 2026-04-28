# Pallas Kernel Demo — Colab v5e-1 (Kernel-Only)

- **Hotspot**: fused RMSNorm + residual add
- **Hidden size**: 2048 (TinyLlama-equivalent)
- **Chosen block_rows**: 512

## 1. Microbench (Pallas vs XLA fused reference)

| tokens | XLA (ms) | Pallas (ms) | speedup |
|---|---|---|---|
| 512 | 0.220 | 0.264 | 0.83× |
| 1024 | 0.274 | 0.269 | 1.02× |
| 2048 | 0.290 | 0.267 | 1.09× |

**Best: 1.09×** (speed gate NOT MET)

## 2. Correctness
- Microbench shapes: 3/3 passed
- End-to-end (incl. odd shapes): 6/6 passed
- Gate: PASSED

## 3. Wrapped-kernel benchmark (with padding path)

| tokens | baseline (ms) | treatment (ms) | speedup |
|---|---|---|---|
| 128 | 0.155 | 0.684 | 0.23× |
| 256 | 0.185 | 0.708 | 0.26× |
| 512 | 0.211 | 0.246 | 0.86× |
| 1024 | 0.223 | 0.247 | 0.90× |
| 2048 | 0.267 | 0.277 | 0.96× |

## 4. Profile op-share diff

_Profile parsing unavailable — open the trace in TensorBoard manually._

## 5. Caveats
- This is a kernel-only demo. End-to-end LLM serving speedup is not measured here.
- The reference is XLA's already-fused RMSNorm+residual; on v6e it's already a strong baseline. Speedups in the 1.1–1.5× range are realistic; >2× would require deeper exploration (different layouts, better pipelining).
- block_rows tuning is per-chip-generation; rerun the sweep on v5e if you switch.

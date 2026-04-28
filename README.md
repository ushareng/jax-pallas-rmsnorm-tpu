# Pallas TPU Kernel Optimization — Colab Demo

A single-notebook walkthrough of a complete kernel optimization workflow on TPU using [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html). Built for the TPU sprint — runs end-to-end on a Colab Pro TPU runtime in 2–3 minutes.

The target: replace XLA's compiled `RMSNorm + residual add` (a real memory-bound bottleneck in Llama-style models) with a hand-written Pallas kernel, and rigorously measure whether it actually wins.

## What's in the notebook

The notebook walks through a **9-stage methodology** for kernel optimization:

1. **Synthetic baseline** — time XLA's compiled fused-RMSNorm reference across realistic shapes
2. **Profile** — JAX trace of the reference, look at op-level cost breakdown
3. **Design** — kernel layout decisions written down before any code (the human checkpoint)
4. **Implement + microbench** — Pallas kernel with `BlockSpec` tiles, correctness gate, speed gate, automatic `block_rows` sweep
5. **Integrate** — wrap the raw kernel as a drop-in callable that handles arbitrary shapes (padding path)
6. **Validate** — multi-shape `allclose` including non-block-multiple sizes
7. **Benchmark** — time the wrapped kernel, diff against Stage 1 baseline
8. **Re-profile** — confirm op-level cost shifted as expected
9. **Report** — generate a markdown summary diffing all stages

Every stage produces JSON output in `/content/results/` so later stages can diff against earlier ones.

## How to run

1. Open `pallas_kernel_TPU.ipynb` in Colab Pro
2. **Runtime → Change runtime type → v5e-1 TPU** (or v6e-1)
3. Run the **first cell** (the JAX downgrade)
4. **Runtime → Restart session** (this is mandatory — pip changes don't apply until restart)
5. Run all remaining cells from the top

Total runtime: ~2–3 minutes.

### Why the JAX pin matters

Colab's default JAX version (currently 0.7.x) emits Mosaic IR at a version newer than Colab's installed `libtpu` can consume, producing `Failed to deserialize the Mosaic module: Unsupported version` errors. The first cell pins to **JAX 0.4.34**, which emits a Mosaic IR version that Colab's libtpu accepts. You'll see some pip dependency-conflict warnings (orbax, flax, optax want a newer JAX) — those are safe to ignore for this demo, which only uses `jax`, `jaxlib`, and `pallas`.

## The kernel

The full kernel is about 20 lines. It fuses two operations that XLA usually compiles into separate kernels with an HBM round-trip in between:

```python
def _rmsnorm_residual_kernel(x_ref, residual_ref, weight_ref,
                             out_ref, new_residual_ref, *, eps):
    x = x_ref[...]
    r = residual_ref[...]
    w = weight_ref[...]
    # 1. Fused residual add — stays in VMEM
    new_r = x + r
    new_residual_ref[...] = new_r
    # 2. RMSNorm in fp32 for numerical stability
    x32 = new_r.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=1, keepdims=True)
    inv = jax.lax.rsqrt(var + eps)
    normed = (x32 * inv).astype(new_r.dtype)
    # 3. Apply weight
    out_ref[...] = normed * w[None, :]
```

Key design choices, all explained in the notebook's Stage 3 markdown:

- **Tile shape** `(block_rows, hidden)` — full row stays in VMEM, swept across `{64, 128, 256, 512}` to find the per-chip sweet spot
- **fp32 accumulator** for the RMS reduction — required for numerical stability with bf16 inputs
- **Single-pass HBM**: read `x`, `residual`, `weight` once; write `output` and `new_residual` once
- **Padding wrapper** for arbitrary token counts — pads to a `block_rows` multiple, slices the output back down

## Results from the saved run

The notebook ships with execution outputs from a v5e-1 run. Headline numbers:

**Microbench (Pallas vs XLA fused reference):**

| tokens | XLA (ms) | Pallas (ms) | speedup |
|---|---|---|---|
| 512 | 0.220 | 0.264 | 0.83× |
| 1024 | 0.274 | 0.269 | 1.02× |
| 2048 | 0.290 | 0.267 | 1.09× |

**Best: 1.09×.** The 1.2× speed gate was **not met** on v5e.

**Correctness:** all gates passed.
- Microbench shapes: 3/3 passed (`allclose` against JAX reference)
- End-to-end including non-block-multiple shapes: 6/6 passed

**Chosen `block_rows`:** 512 (after sweep at tokens=1024).

## What the results actually mean

The honest reading: on v5e, **XLA's compiled fused-RMSNorm is already very good**. The compiler's existing fusion captures most of the available memory-bandwidth savings, leaving little room for a hand-written kernel to win on this specific op at this specific scale.

This is a useful, not disappointing, result. A few things it tells us:

- The methodology worked exactly as designed — the speed gate caught a kernel that would have been an end-to-end regression if naively integrated. **A correct-but-slower kernel is the worst possible outcome**, and the gate prevented that.
- The win cases for hand-written kernels are typically (a) ops XLA doesn't fuse well (paged attention, complex sampling), (b) larger sequence lengths where memory-bandwidth headroom dominates, or (c) chips with different VMEM characteristics. v6e (Trillium) tends to show better speedups on this op than v5e.
- Different hotspots are likely to be more rewarding targets on v5e — fused attention, RoPE, and sampling are typical candidates.

## What the notebook demonstrates

Even though the speedup didn't clear the 1.2× gate on v5e, the notebook is a complete, working example of:

- **Pallas kernel mechanics** — `pallas_call`, `BlockSpec`, grid, VMEM tiles, fp32 accumulator
- **Empirical block-size tuning** rather than guessing — sweep at runtime, pick the winner
- **Correctness-first methodology** — `allclose` against a JAX reference before timing anything
- **Honest measurement** — fixed shapes, fixed seeds, repeats with stdev, before/after diff
- **Padding glue** for arbitrary shapes — real callers don't always send block-aligned tensors
- **Stage-by-stage gates** — speed gate, correctness gate, with explicit pass/fail in the report

The methodology transfers directly to other kernels and other hardware. To target a different hotspot, only the kernel body changes; the surrounding pipeline stays the same.

## Hardware notes

| Chip | Status | Notes |
|---|---|---|
| TPU v5e-1 | ✅ Tested, runs end-to-end | XLA's fused-RMSNorm is strong; expect ~1.0–1.1× on this op |
| TPU v6e-1 | ✅ Should run end-to-end | Likely higher speedup; needs `block_rows` re-tuning (sweep is automatic) |

The notebook auto-detects the chip and runs the `block_rows` sweep on whichever hardware it's on.

## File outputs

After running, `/content/results/` contains:

- `01_baseline.json` through `08_reprofile.json` — per-stage measurements
- `REPORT.md` — the rendered final summary
- `profiles/baseline/` and `profiles/treatment/` — XLA traces (download with `!tar -czf trace.tar.gz /content/results/profiles` to inspect locally in TensorBoard)

Results live in the runtime only and are lost on disconnect.

## License

MIT.

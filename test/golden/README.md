# Golden Output Tests

Golden output tests verify correctness after refactoring by comparing outputs
against a known-good baseline. RTOD is deterministic given the same inputs, so
any diff against the golden file indicates a regression.

## Setup (run once on Linux GPU machine)

```bash
./build/bin/outlier_detection \
    --n 10000 --R 0.028 --K 50 --window 1000 --slide 500 \
    -f data/gaussian.txt \
    > test/golden/gaussian_baseline.log 2>&1

./build/bin/outlier_detection \
    --n 10000 --R 0.45 --K 50 --window 1000 --slide 500 \
    -f data/stock.txt \
    > test/golden/stock_baseline.log 2>&1
```

## Verify (after any refactor)

```bash
./build/bin/outlier_detection \
    --n 10000 --R 0.028 --K 50 --window 1000 --slide 500 \
    -f data/gaussian.txt 2>&1 \
    | diff test/golden/gaussian_baseline.log -

./build/bin/outlier_detection \
    --n 10000 --R 0.45 --K 50 --window 1000 --slide 500 \
    -f data/stock.txt 2>&1 \
    | diff test/golden/stock_baseline.log -
```

## Important

- Golden outputs MUST be regenerated after any change that alters numerical results
  (e.g., changing the distance formula, R value, or BVH build flags).
- Golden outputs are architecture-dependent (GPU model, CUDA version, OptiX version).
  Always regenerate on the target hardware before comparing.
- The `[Mem]` and `[Time]` lines are hardware-dependent; strip them before diffing
  if you only care about correctness:
  ```bash
  grep -v -E '^\[(Mem|Time)' output.log | diff golden.log -
  ```

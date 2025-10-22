# Sumcheck Experiment Tool

A command-line tool for experimenting with different sumcheck implementations using the actual Jolt `SingleSumcheck` abstraction. Threading defaults to rayon's global pool, and you can override the thread count via CLI flags.

## Usage

```bash
# Run from the jolt root directory, always in release mode!
cargo run --release -p jolt-core --bin bench -- [COMMAND] [OPTIONS]
```

## Commands

### Run Single Experiment
```bash
cargo run --release -p jolt-core --bin bench -- run --T 10 --d 2 --mode 0 --tile-len 256
```

**Options:**
- `-T, --T <T>`: Size of sumcheck (2^T), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2  
- `-m, --mode <MODE>`: Mode to run (0=Batch, 1=Tiling), default: 0
- `--tile-len <N>`: Tile length (pairs per tile) for tiling mode. If omitted, a heuristic is used.
- `--threads <N>`: Override rayon threads for this run (e.g., `--threads 16`).

### Compare Implementations
```bash
cargo run --release -p jolt-core --bin bench -- compare --T 10 --d 2 --tile-len 256
```

**Options:**
- `-T, --T <T>`: Size of sumcheck (2^T), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2
- `--tile-len <N>`: Tile length (pairs per tile) for tiling mode. If omitted, a heuristic is used.
- `--threads <N>`: Override rayon threads for this run (e.g., `--threads 16`).

### Batch Experiments
Run a grid of experiments over T and d:
```bash
cargo run --release -p jolt-core --bin bench -- batch --T 15,20,24 --d 2,3,4 --tile-len 128,256,512 --threads 8,16,32
```

Batch mode iterates over each thread count and runs the full grid for every `T × d`. Logs include the selected `threads` value for each run. The chart now separates phases: it draws four series per `(d, threads)` pair — `batch first_sum`, `batch prove`, `tiling first_sum`, and `tiling prove` — so you can see phase-level scaling.

## Modes

- **Mode 0 - Batch**: Uses `SingleSumcheck` with `ProductSumcheck` in batch mode; input_claim uses a parallel map-reduce over i (no tiling), mirroring baseline behavior.
- **Mode 1 - Tiling**: Uses `SingleSumcheck` with `ProductSumcheck` in tiling mode; input_claim uses a partition-friendly tiled fold/reduce with `--tile-len` if provided, or an internal heuristic when omitted.

## Implementation Details

All modes use the actual Jolt `SingleSumcheck` abstraction:
- **SingleSumcheck::prove()**: Real sumcheck proving using the Jolt implementation
- **SumcheckInstance Trait**: Proper implementation of the sumcheck instance interface
- **DensePolynomial**: Real multilinear polynomial implementation from Jolt
- **Blake2bTranscript**: Actual transcript implementation for challenge generation
- **PolynomialEvaluation**: Proper sumcheck evaluation methods with `sumcheck_evals()`
- **PolynomialBinding**: Correct polynomial binding with `bind_parallel()`
- **Rayon Parallelization**: Parallel processing using rayon (global pool); CLI flags can override thread count per run or per batch variant
- **Challenge Handling**: Proper `F::Challenge` type conversion with `MontU128Challenge`

## Random Data Generation

The tool uses deterministic pseudo-random data generation with seed "fun":
- Polynomial coefficients are generated using a fast SplitMix64-style mixer
- This ensures reproducible results across runs
- Different modes use the same random data for fair comparison

## Tiling

- Tiling uses tiled evaluation to improve cache locality. You can control the tile length directly with `--tile-len` (pairs per tile). If not provided, a reasonable heuristic is used.
- Deferred binding: bind() records the next challenge; at the start of compute, the instance applies the pending challenge to produce halved polynomials, then computes the product-sum evaluations.
- Big-endian normalization is used for opening points and final claim evaluation, consistent with the codebase.

## Output

The tool prints timing information including:
- Threads used for the run
- Total time and per-phase times
- For Compare: total, `first_sum` and `prove` times for Batch and Tiling, per-thread throughput, and speedup (Batch/Tiling)
- For Batch: per-run `first_sum` and `prove` times for both implementations, with the `threads` value included in each log line; the generated chart shows separate series for each `(d, threads)` pair

## Examples
```bash
# Single run
cargo run --release -p jolt-core --bin bench -- run --T 18 --d 3 --mode 1 --tile-len 256

# Compare batch vs tiling for a single setting
cargo run --release -p jolt-core --bin bench -- compare --T 24 --d 2 --tile-len 256 --threads 16

# Batch grid
cargo run --release -p jolt-core --bin bench -- batch --T 20,22 --d 2,3 --tile-len 128,256 --threads 8,16
```

# Sumcheck Experiment Tool

A command-line tool for experimenting with different sumcheck implementations using the actual Jolt `SingleSumcheck` abstraction and measuring their performance with configurable threading.

## Usage

```bash
# Run from the jolt root directory
cargo run -p jolt-core --bin bench -- [COMMAND] [OPTIONS]
```

## Commands

### Run Single Experiment
```bash
cargo run -p jolt-core --bin bench -- run --k 10 --d 2 --mode 0 --iterations 5 --threads 4
```

**Options:**
- `-k, --k <K>`: Size of sumcheck (2^k), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2  
- `-m, --mode <MODE>`: Mode to run (0-2), default: 0
- `-i, --iterations <ITERATIONS>`: Number of iterations, default: 1
- `-t, --threads <THREADS>`: Number of threads (0 = auto-detect), default: 0

### Compare Implementations
```bash
cargo run -p jolt-core --bin bench -- compare --k 10 --d 2 --iterations 5 --threads 4
```

**Options:**
- `-k, --k <K>`: Size of sumcheck (2^k), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2
- `-i, --iterations <ITERATIONS>`: Iterations per mode, default: 5
- `-t, --threads <THREADS>`: Number of threads (0 = auto-detect), default: 0

### Batch Experiments
Run a grid of experiments over k, d, and threads:
```bash
cargo run -p jolt-core --bin bench -- batch --k 15,20,24 --d 2,3,4 --threads 4,8,12 --iterations 2
```

## Modes

- **Mode 0 - Baseline (SingleSumcheck)**: Uses `SingleSumcheck` with `ProductSumcheck` (configurable degree d)

## Implementation Details

All modes use the actual Jolt `SingleSumcheck` abstraction:
- **SingleSumcheck::prove()**: Real sumcheck proving using the Jolt implementation
- **SumcheckInstance Trait**: Proper implementation of the sumcheck instance interface
- **DensePolynomial**: Real multilinear polynomial implementation from Jolt
- **Blake2bTranscript**: Actual transcript implementation for challenge generation
- **PolynomialEvaluation**: Proper sumcheck evaluation methods with `sumcheck_evals()`
- **PolynomialBinding**: Correct polynomial binding with `bind_parallel()`
- **Rayon Parallelization**: Real parallel processing using rayon
- **Challenge Handling**: Proper `F::Challenge` type conversion with `MontU128Challenge`

## Random Data Generation

The tool uses deterministic pseudo-random data generation with seed "fun":
- Polynomial coefficients are generated using `StdRng` with the seed
- This ensures reproducible results across runs
- Different modes use the same random data for fair comparison

## Examples

```bash
# Quick test with simple sumcheck
cargo run -p jolt-core --bin sumcheck_experiment -- run --k 8 --d 2 --mode 0 --threads 4

# Test multi-product sumcheck with degree 3
cargo run -p jolt-core --bin sumcheck_experiment -- run --k 8 --d 3 --mode 1 --threads 4

# Compare all modes with threading
cargo run -p jolt-core --bin sumcheck_experiment -- compare --k 10 --d 3 --iterations 10 --threads 8

# Test with auto-detected threads
cargo run -p jolt-core --bin sumcheck_experiment -- run --k 12 --d 2 --mode 2
```

## Threading & Tiling

The tool supports configurable threading:
- Set `--threads 0` (default) to auto-detect the number of CPU cores
- Set `--threads N` to use exactly N threads
- All modes use parallel processing with rayon for polynomial operations.
- Baseline uses contiguous memory traversal and a dedicated threadpool when `--threads > 0`.
- Sliced uses tiled evaluation to improve cache locality. Tile length is auto-tuned to fit L1 cache:
  - `tile_len ≈ L1_bytes / (degree * elem_bytes)`, clamped to [64, 1024] and rounded to a power of two.
  - Binding uses HighToLow to match Jolt’s sumcheck pairing `[2*j], [2*j+1]`.
  - Tiled loops precompute products for h(0) and h(t) across all t in a single pass.

## Output

The tool provides detailed timing information including:
- Thread configuration used
- Total execution time
- Average time per iteration
- Minimum and maximum times
- Success/failure status and explicit proof verification (both baseline and sliced).
- Baseline vs Sliced timing with avg/med/min/max and a speedup factor (Baseline/Sliced).

## Credibility

This tool uses the actual Jolt `SingleSumcheck` abstraction and infrastructure:
- Real `SingleSumcheck::prove()` and `SingleSumcheck::verify()` methods
- Proper `SumcheckInstance` trait implementation
- Authentic polynomial evaluation and binding methods from the Jolt codebase
- Real transcript implementation for challenge generation
- Correct field arithmetic and challenge types

This makes the experiments highly credible and representative of actual Jolt sumcheck performance characteristics.

## Examples
```bash
# Single run
cargo run -p jolt-core --bin bench -- run --k 18 --d 3 --mode 0 --iterations 2 --threads 8

# Compare baseline vs sliced for a single setting
cargo run -p jolt-core --bin bench -- compare --k 18 --d 3 --iterations 3 --threads 8

# Batch grid
cargo run -p jolt-core --bin bench -- batch --k 15,20,24 --d 2,3,4 --threads 4,8,12 --iterations 2
```

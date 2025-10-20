# Sumcheck Experiment Tool

A command-line tool for experimenting with different sumcheck implementations using the actual Jolt `SingleSumcheck` abstraction. Threading is auto-managed by rayon.

## Usage

```bash
# Run from the jolt root directory
cargo run -p jolt-core --bin bench -- [COMMAND] [OPTIONS]
```

## Commands

### Run Single Experiment
```bash
cargo run -p jolt-core --bin bench -- run --T 10 --d 2 --mode 0 --l1-kb 64
```

**Options:**
- `-T, --T <T>`: Size of sumcheck (2^T), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2  
- `-m, --mode <MODE>`: Mode to run (0=Batch, 1=Streaming), default: 0
- `--l1-kb <KB>`: L1 data cache size in kB for tile sizing, default: 32
Threads are auto-managed by rayon; the `--threads` flag is deprecated.

### Compare Implementations
```bash
cargo run -p jolt-core --bin bench -- compare --T 10 --d 2 --l1-kb 64
```

**Options:**
- `-T, --T <T>`: Size of sumcheck (2^T), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2
- `-i, --iterations <ITERATIONS>`: Iterations per mode, default: 5
Threads are auto-managed by rayon; the `--threads` flag is deprecated.

### Batch Experiments
Run a grid of experiments over T and d:
```bash
cargo run -p jolt-core --bin bench -- batch --T 15,20,24 --d 2,3,4 --l1-kb 64
```

## Modes

- **Mode 0 - Batch**: Uses `SingleSumcheck` with `ProductSumcheck` in batch mode; input_claim uses a parallel map-reduce over i (no tiling), mirroring baseline behavior.
- **Mode 1 - Streaming**: Uses `SingleSumcheck` with `ProductSumcheck` in streaming mode; input_claim uses a partition-friendly tiled fold/reduce with tile_len derived from `--l1-kb`.

## Implementation Details

All modes use the actual Jolt `SingleSumcheck` abstraction:
- **SingleSumcheck::prove()**: Real sumcheck proving using the Jolt implementation
- **SumcheckInstance Trait**: Proper implementation of the sumcheck instance interface
- **DensePolynomial**: Real multilinear polynomial implementation from Jolt
- **Blake2bTranscript**: Actual transcript implementation for challenge generation
- **PolynomialEvaluation**: Proper sumcheck evaluation methods with `sumcheck_evals()`
- **PolynomialBinding**: Correct polynomial binding with `bind_parallel()`
- **Rayon Parallelization**: Parallel processing using rayon (global pool)
- **Challenge Handling**: Proper `F::Challenge` type conversion with `MontU128Challenge`

## Random Data Generation

The tool uses deterministic pseudo-random data generation with seed "fun":
- Polynomial coefficients are generated using `StdRng` with the seed
- This ensures reproducible results across runs
- Different modes use the same random data for fair comparison

## Tiling and L1 sizing

- Streaming uses tiled evaluation to improve cache locality. Tile length is derived from the configured L1 size:
  - `tile_len ≈ L1_bytes / (degree * elem_bytes)`, clamped to [64, 1024] and rounded to a power of two.
  - `--l1-kb` sets L1_bytes = KB * 1024; default is 32 KiB.
  - Deferred binding: bind() records the next challenge; at the start of compute, the instance applies the pending challenge to produce halved polynomials, then computes the product-sum evaluations.
  - Big-endian normalization is used for opening points and final claim evaluation, consistent with the codebase.

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
cargo run -p jolt-core --bin bench -- run --T 18 --d 3 --mode 0 --l1-kb 64

# Compare batch vs streaming for a single setting
cargo run -p jolt-core --bin bench -- compare --T 15 --d 2 --l1-kb 64

# Batch grid
cargo run -p jolt-core --bin bench -- batch --T 15,20,24 --d 2,3,4 --l1-kb 64
```

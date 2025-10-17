# Sumcheck Experiment Tool

A command-line tool for experimenting with different sumcheck implementations using the actual Jolt `SingleSumcheck` abstraction and measuring their performance with configurable threading.

## Usage

```bash
# Run from the jolt root directory
cargo run -p jolt-core --bin sumcheck_experiment -- [COMMAND] [OPTIONS]
```

## Commands

### Run Single Experiment
```bash
cargo run -p jolt-core --bin sumcheck_experiment -- run --k 10 --d 2 --mode 1 --iterations 5 --threads 4
```

**Options:**
- `-k, --k <K>`: Size of sumcheck (2^k), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2  
- `-m, --mode <MODE>`: Mode to run (0-2), default: 0
- `-i, --iterations <ITERATIONS>`: Number of iterations, default: 1
- `-t, --threads <THREADS>`: Number of threads (0 = auto-detect), default: 0

### Compare Implementations
```bash
cargo run -p jolt-core --bin sumcheck_experiment -- compare --k 10 --d 2 --iterations 5 --threads 4
```

**Options:**
- `-k, --k <K>`: Size of sumcheck (2^k), default: 10
- `-d, --d <D>`: Degree of polynomial, default: 2
- `-i, --iterations <ITERATIONS>`: Iterations per mode, default: 5
- `-t, --threads <THREADS>`: Number of threads (0 = auto-detect), default: 0

## Modes

- **Mode 0 - Simple Sumcheck**: Uses `SingleSumcheck` with a basic `SumcheckInstance` implementation (degree 2)
- **Mode 1 - Multi-Product Sumcheck**: Uses `SingleSumcheck` with a multi-product `SumcheckInstance` (configurable degree d)
- **Mode 2 - Multi-Product Sumcheck**: Same as Mode 1, for comparison

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

## Threading

The tool supports configurable threading:
- Set `--threads 0` (default) to auto-detect the number of CPU cores
- Set `--threads N` to use exactly N threads
- All modes use parallel processing with rayon for polynomial operations
- Different parallelization strategies are used across modes for comparison

## Output

The tool provides detailed timing information including:
- Thread configuration used
- Total execution time
- Average time per iteration
- Minimum and maximum times
- Success/failure status
- Performance comparison rankings

## Credibility

This tool uses the actual Jolt `SingleSumcheck` abstraction and infrastructure:
- Real `SingleSumcheck::prove()` and `SingleSumcheck::verify()` methods
- Proper `SumcheckInstance` trait implementation
- Authentic polynomial evaluation and binding methods from the Jolt codebase
- Real transcript implementation for challenge generation
- Correct field arithmetic and challenge types

This makes the experiments highly credible and representative of actual Jolt sumcheck performance characteristics.

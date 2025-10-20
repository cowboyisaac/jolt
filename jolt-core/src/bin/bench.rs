use clap::{Parser, Subcommand};
use ark_bn254::Fr;
use std::time::Instant;

mod product_sumcheck;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::subprotocols::sumcheck::{SingleSumcheck, SumcheckInstance};
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use product_sumcheck::{ProductSumcheck, ExecutionMode};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
// verification for benches removed; covered by tests

#[derive(Parser)]
#[command(name = "bench")]
#[command(about = "A tool for benchmarking sumcheck implementations and performance")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run sumcheck experiments with different configurations
    Run {
        /// Size of the sumcheck (2^T)
        #[arg(short = 'T', long = "T", default_value = "10")]
        t: u32,
        
        /// Degree of the polynomial
        #[arg(short, long, default_value = "2")]
        d: u32,
        
        /// Mode to run (0=Batch, 1=Streaming)
        #[arg(short, long, default_value = "0")]
        mode: u32,

        /// L1 data cache size in kB for tile sizing (streaming). Default 32.
        #[arg(long = "l1-kb", default_value = "32")]
        l1_kb: usize,
    },
    /// Compare different sumcheck implementations
    Compare {
        /// Size of the sumcheck (2^T)
        #[arg(short = 'T', long = "T", default_value = "10")]
        t: u32,
        
        /// Degree of the polynomial
        #[arg(short, long, default_value = "2")]
        d: u32,
        /// L1 data cache size in kB for tile sizing (streaming). Default 32.
        #[arg(long = "l1-kb", default_value = "32")]
        l1_kb: usize,
    },
    /// Batch compare across grids of T, d, threads
    Batch {
        /// Sizes (2^T). Comma-separated list, e.g. 15,20,24
        #[arg(long = "T", value_delimiter = ',', required = true)]
        t_list: Vec<u32>,
        /// Degrees d. Comma-separated list, e.g. 2,3,4
        #[arg(long = "d", value_delimiter = ',', required = true)]
        d_list: Vec<u32>,
        /// Deprecated. Threads are auto-managed by rayon now.
        #[arg(long = "threads", value_delimiter = ',', required = false)]
        threads_list: Vec<usize>,
        /// L1 data cache size in kB for tile sizing (streaming). Default 32.
        #[arg(long = "l1-kb", default_value = "32")]
        l1_kb: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run { t, d, mode, l1_kb } => run_single_experiment(t, d, mode, l1_kb),
        Commands::Compare { t, d, l1_kb } => {
            match compare_implementations(t, d, l1_kb) {
                Ok(()) => println!("\nCompare finished successfully."),
                Err(e) => eprintln!("\nCompare failed: {}", e),
            }
        }
        Commands::Batch { t_list, d_list, threads_list, l1_kb } => {
            run_batch_experiments(t_list, d_list, threads_list, l1_kb);
        }
    }
}

// Threads are auto-managed by rayon now.

fn run_single_experiment(t: u32, d: u32, mode: u32, l1_kb: usize) {
    println!("Running sumcheck experiment: T={}, d={}, mode={}", t, d, mode);
    // Only profile the proving time, not verification
    let start = Instant::now();
    let result = match mode {
        0 => run_batch_sumcheck::<Fr>(t, d, l1_kb),
        1 => run_streaming_sumcheck::<Fr>(t, d, l1_kb),
        _ => {
            println!("Unknown mode {}, falling back to mode 0", mode);
            run_batch_sumcheck::<Fr>(t, d, l1_kb)
        }
    };
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(output) => println!("Completed in {:.2}ms, output: {}", duration_ms, output),
        Err(e) => println!("Run failed: {}", e),
    }
}

fn compare_implementations(t: u32, d: u32, l1_kb: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing sumcheck implementations: T={}, d={}", t, d);
    let polys = build_random_dense_polys::<Fr>(t, d, "fun");
    let (claim_batch, t_batch) = timed_batch::<Fr>(polys.clone(), l1_kb);
    let (claim_streaming, t_streaming) = timed_streaming_with_polys::<Fr>(polys, l1_kb);
    println!("Batch={:.2}ms, Streaming={:.2}ms, claim={} (equal: {})",
        t_batch, t_streaming, claim_batch, claim_batch == claim_streaming);
    println!("Speedup (Batch/Streaming): {:.2}x", t_batch / t_streaming);
    Ok(())
}

fn run_batch_experiments(t_list: Vec<u32>, d_list: Vec<u32>, threads_list: Vec<usize>, l1_kb: usize) {
    println!("Batch experiments: Ts={:?}, ds={:?}, threads={:?}", t_list, d_list, threads_list);
    for &t in &t_list {
        for &d in &d_list {
            if threads_list.is_empty() {
                println!("\n== Case: T={}, d={}, threads={} ==", t, d, rayon::current_num_threads());
                if let Err(e) = compare_implementations(t, d, l1_kb) {
                    println!("  ERROR: {}", e);
                }
            } else {
                for threads in threads_list.iter().cloned() {
                    if threads > 0 { let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global(); }
                    println!("\n== Case: T={}, d={}, threads={} ==", t, d, if threads==0 { rayon::current_num_threads() } else { threads });
                    if let Err(e) = compare_implementations(t, d, l1_kb) {
                        println!("  ERROR: {}", e);
                    }
                }
            }
        }
    }
}

// ------------ Helpers moved from impl -------------

fn build_random_dense_polys<F: JoltField>(t: u32, d: u32, seed: &str) -> Vec<DensePolynomial<F>> {
    let n = 1 << t;
    let degree = d as usize;
    let mut polynomials = Vec::with_capacity(degree);
    for poly_idx in 0..degree {
        let mut coeffs = vec![F::zero(); n];
        let poly_seed = seed.as_bytes().iter().map(|&b| b as u64).sum::<u64>() + poly_idx as u64;
        let mut poly_rng = StdRng::seed_from_u64(poly_seed);
        for i in 0..n { coeffs[i] = F::from_u64(poly_rng.gen_range(1..1000)); }
        polynomials.push(DensePolynomial::new(coeffs));
    }
    polynomials
}

fn run_batch_sumcheck<F: JoltField>(t: u32, d: u32, l1_kb: usize) -> Result<F, Box<dyn std::error::Error>> {
    let polys = build_random_dense_polys::<F>(t, d, "fun");
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Batch, Some(l1_kb));
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_proof, _chals) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    Ok(<ProductSumcheck<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck))
}

fn run_streaming_sumcheck<F: JoltField>(t: u32, d: u32, l1_kb: usize) -> Result<F, Box<dyn std::error::Error>> {
    let polys = build_random_dense_polys::<F>(t, d, "fun");
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Streaming, Some(l1_kb));
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_proof, _chals) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    Ok(<ProductSumcheck<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck))
}

fn timed_batch<F: JoltField>(polys: Vec<DensePolynomial<F>>, l1_kb: usize) -> (F, f64) {
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Batch, Some(l1_kb));
    let start = Instant::now();
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_p,_c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let elapsed = start.elapsed().as_secs_f64()*1000.0;
    (sumcheck.input_claim, elapsed)
}

fn timed_streaming_with_polys<F: JoltField>(polys: Vec<DensePolynomial<F>>, l1_kb: usize) -> (F, f64) {
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Streaming, Some(l1_kb));
    let start = Instant::now();
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_p,_c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let elapsed = start.elapsed().as_secs_f64()*1000.0;
    (sumcheck.input_claim, elapsed)
}

// removed verify helpers from bench
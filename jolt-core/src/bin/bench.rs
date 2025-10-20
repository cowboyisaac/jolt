use clap::{Parser, Subcommand};
use ark_bn254::Fr;
use std::time::Instant;
use rayon::prelude::*;

mod product_sumcheck;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::subprotocols::sumcheck::{SingleSumcheck, SumcheckInstance};
use jolt_core::transcripts::{Blake2bTranscript, Transcript};
use product_sumcheck::{ProductSumcheck, ExecutionMode};
// RNG imports removed; we use a custom splitmix64 for faster deterministic generation
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
    let total_start = Instant::now();
    let gen_start = Instant::now();
    let polys = build_random_dense_polys::<Fr>(t, d, "fun");
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    let (claim, prove_ms) = match mode {
        0 => timed_batch::<Fr>(polys, l1_kb),
        1 => timed_streaming_with_polys::<Fr>(polys, l1_kb),
        _ => {
            println!("Unknown mode {}, falling back to mode 0", mode);
            timed_batch::<Fr>(build_random_dense_polys::<Fr>(t, d, "fun"), l1_kb)
        }
    };
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    println!("Completed: total={:.2}ms, gen={:.2}ms, prove={:.2}ms, output={}", total_ms, gen_ms, prove_ms, claim);
}

fn compare_implementations(t: u32, d: u32, l1_kb: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing sumcheck implementations: T={}, d={}", t, d);
    let overall_start = Instant::now();
    let gen_start = Instant::now();
    let polys = build_random_dense_polys::<Fr>(t, d, "fun");
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

    let clone_start = Instant::now();
    let polys_clone = polys.clone();
    let clone_ms = clone_start.elapsed().as_secs_f64() * 1000.0;

    let (claim_batch, t_batch) = timed_batch::<Fr>(polys_clone, l1_kb);
    let (claim_streaming, t_streaming) = timed_streaming_with_polys::<Fr>(polys, l1_kb);

    let overall_ms = overall_start.elapsed().as_secs_f64() * 1000.0;
    let accounted = gen_ms + clone_ms + t_batch + t_streaming;
    let overhead_ms = (overall_ms - accounted).max(0.0);

    println!("Batch={:.2}ms, Streaming={:.2}ms, claim={} (equal: {})",
        t_batch, t_streaming, claim_batch, claim_batch == claim_streaming);
    println!("Total={:.2}ms (gen={:.2}ms, clone={:.2}ms, batch={:.2}ms, streaming={:.2}ms, overhead={:.2}ms)",
        overall_ms, gen_ms, clone_ms, t_batch, t_streaming, overhead_ms);
    println!("Speedup (Batch/Streaming proving time ONLY): {:.2}x", t_batch / t_streaming);
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
    // Deterministic, fast, parallel data generation using splitmix64-style mixing.
    #[inline]
    fn mix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    let n = 1 << t;
    let degree = d as usize;
    let base_seed = seed.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));

    (0..degree)
        .into_par_iter()
        .map(|poly_idx| {
            let mut coeffs = vec![F::zero(); n];
            let poly_seed = base_seed.wrapping_add(poly_idx as u64);
            // Fill in parallel chunks for better cache and parallelism on large n
            coeffs
                .par_chunks_mut(1024)
                .enumerate()
                .for_each(|(chunk_i, chunk)| {
                    let start = chunk_i * 1024;
                    for (off, c) in chunk.iter_mut().enumerate() {
                        let i = start + off;
                        let r = mix64(poly_seed ^ (i as u64).wrapping_mul(0x9E3779B185EBCA87));
                        // Map to [1, 999]
                        let val = 1 + (r % 999);
                        *c = F::from_u64(val);
                    }
                });
            DensePolynomial::new(coeffs)
        })
        .collect()
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
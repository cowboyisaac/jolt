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
use std::cell::RefCell;
use std::rc::Rc;
use jolt_core::poly::opening_proof::VerifierOpeningAccumulator;

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
        
        /// Mode to run (0=baseline)
        #[arg(short, long, default_value = "0")]
        mode: u32,
        
        /// Number of iterations to run
        #[arg(short, long, default_value = "1")]
        iterations: usize,
        
        /// Deprecated. Threads are auto-managed by rayon now.
        #[arg(short, long, default_value = "0")]
        _threads: usize,
    },
    /// Compare different sumcheck implementations
    Compare {
        /// Size of the sumcheck (2^T)
        #[arg(short = 'T', long = "T", default_value = "10")]
        t: u32,
        
        /// Degree of the polynomial
        #[arg(short, long, default_value = "2")]
        d: u32,
        
        /// Number of iterations per mode
        #[arg(short, long, default_value = "5")]
        iterations: usize,
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
        /// Iterations per setting
        #[arg(short, long, default_value = "2")]
        iterations: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run {
            t,
            d,
            mode,
            iterations,
            _threads,
        } => {
            run_single_experiment(t, d, mode, iterations);
        }
        Commands::Compare {
            t,
            d,
            iterations,
        } => {
            match compare_implementations(t, d, iterations) {
                Ok(()) => println!("\nCompare finished successfully."),
                Err(e) => eprintln!("\nCompare failed: {}", e),
            }
        }
        Commands::Batch { t_list, d_list, threads_list, iterations } => {
            run_batch_experiments(t_list, d_list, threads_list, iterations);
        }
    }
}

// Threads are auto-managed by rayon now.

fn run_single_experiment(t: u32, d: u32, mode: u32, iterations: usize) {
    println!("Running sumcheck experiment: T={}, d={}, mode={}, iterations={}", t, d, mode, iterations);
    
    let mut times = Vec::new();
    let mut success = true;
    let mut _error_msg = None;
    
    for i in 0..iterations {
        println!("Iteration {}/{}", i + 1, iterations);
        
        // Only profile the proving time, not verification
        let start = Instant::now();
        let result = match mode {
            0 => run_baseline_sumcheck::<Fr>(t, d),
            1 => run_streaming_sumcheck::<Fr>(t, d),
            _ => {
                println!("Unknown mode {}, falling back to mode 0", mode);
                run_baseline_sumcheck::<Fr>(t, d)
            }
        };
        let duration = start.elapsed();
        
        match result {
            Ok(output) => {
                times.push(duration.as_secs_f64() * 1000.0);
                println!("Iteration {} completed in {:.2}ms, output: {}", 
                          i + 1, duration.as_secs_f64() * 1000.0, output);
            }
            Err(e) => {
                println!("Iteration {} failed: {}", i + 1, e);
                success = false;
                _error_msg = Some(e.to_string());
                break;
            }
        }
    }
    
    if !times.is_empty() {
        let total_time = times.iter().sum::<f64>();
        let avg_time = total_time / times.len() as f64;
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(0.0, f64::max);
        
        println!("\nExperiment completed:");
        let mode_name = match mode {
            0 => "Baseline (Batch)",
            1 => "Streaming (Tiled)",
            _ => "Unknown",
        };
        println!("  Mode: {} ({})", mode, mode_name);
        println!("  Size: 2^{} = {}", t, 1 << t);
        println!("  Degree: {}", d);
        println!("  Iterations: {}", times.len());
        println!("  Total time: {:.2}ms", total_time);
        println!("  Average time: {:.2}ms", avg_time);
        println!("  Min time: {:.2}ms", min_time);
        println!("  Max time: {:.2}ms", max_time);
        println!("  Success: {}", success);
    }
}

fn compare_implementations(t: u32, d: u32, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing sumcheck implementations: T={}, d={}, iterations={}", t, d, iterations);

    let mut baseline_times = Vec::new();
    let mut streaming_times = Vec::new();

    for i in 0..iterations {
    let polys = build_random_dense_polys::<Fr>(t, d, "fun");
    let (claim_baseline, t_base) = timed_baseline::<Fr>(polys.clone());
    let (claim_streaming, t_streaming) = timed_streaming_with_polys::<Fr>(polys);
        println!(
            "  Iteration {:>2}/{}: baseline={:.2}ms, streaming={:.2}ms, claim={}",
            i+1, iterations, t_base, t_streaming, claim_baseline
        );
    // Verify proofs for both implementations
        // verify_latest::<Fr>(t, d)?;
        assert_eq!(claim_baseline, claim_streaming, "Claims differ between baseline and streaming");
        baseline_times.push(t_base);
        streaming_times.push(t_streaming);
    }

    let stats = |v: &Vec<f64>| {
        let mut w = v.clone(); w.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = w.len();
        let avg = w.iter().sum::<f64>() / (n as f64);
        let med = if n==0 {0.0} else { w[n/2] };
        let min = w.first().cloned().unwrap_or(0.0);
        let max = w.last().cloned().unwrap_or(0.0);
        (avg, med, min, max)
    };
    let (b_avg,b_med,b_min,b_max) = stats(&baseline_times);
    let (s_avg,s_med,s_min,s_max) = stats(&streaming_times);

    println!("\nThreads: rayon_current={}", rayon::current_num_threads());
    println!("Problem: n=2^{}, degree={} polys", t, d);
    println!("Baseline: avg={:.2}ms, med={:.2}ms, min={:.2}ms, max={:.2}ms", b_avg,b_med,b_min,b_max);
    println!("Streaming:   avg={:.2}ms, med={:.2}ms, min={:.2}ms, max={:.2}ms", s_avg,s_med,s_min,s_max);
    if s_avg>0.0 { println!("Speedup (Baseline/Streaming): {:.2}x", b_avg/s_avg); }
    Ok(())
}

fn run_batch_experiments(t_list: Vec<u32>, d_list: Vec<u32>, threads_list: Vec<usize>, iterations: usize) {
    println!("Batch experiments: Ts={:?}, ds={:?}, threads={:?}, iterations={}", t_list, d_list, threads_list, iterations);
    for &t in &t_list {
        for &d in &d_list {
            if threads_list.is_empty() {
                println!("\n== Case: T={}, d={}, threads={} ==", t, d, rayon::current_num_threads());
                if let Err(e) = compare_implementations(t, d, iterations) {
                    println!("  ERROR: {}", e);
                }
            } else {
                for threads in threads_list.iter().cloned() {
                    if threads > 0 { let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global(); }
                    println!("\n== Case: T={}, d={}, threads={} ==", t, d, if threads==0 { rayon::current_num_threads() } else { threads });
                    if let Err(e) = compare_implementations(t, d, iterations) {
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

fn run_baseline_sumcheck<F: JoltField>(t: u32, d: u32) -> Result<F, Box<dyn std::error::Error>> {
    let polys = build_random_dense_polys::<F>(t, d, "fun");
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Batch);
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_proof, _chals) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let mut vt = Blake2bTranscript::new(b"sumcheck_experiment");
    let opening_acc = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));
    let verify = SingleSumcheck::verify::<F, Blake2bTranscript>(&sumcheck, &_proof, Some(opening_acc), &mut vt);
    assert!(verify.is_ok(), "Sumcheck verification failed");
    Ok(<ProductSumcheck<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck))
}

fn run_streaming_sumcheck<F: JoltField>(t: u32, d: u32) -> Result<F, Box<dyn std::error::Error>> {
    let polys = build_random_dense_polys::<F>(t, d, "fun");
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Streaming);
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_proof, _chals) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let mut vt = Blake2bTranscript::new(b"sumcheck_experiment");
    let opening_acc = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));
    let verify = SingleSumcheck::verify::<F, Blake2bTranscript>(&sumcheck, &_proof, Some(opening_acc), &mut vt);
    assert!(verify.is_ok(), "Streaming sumcheck verification failed");
    Ok(<ProductSumcheck<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck))
}

fn timed_baseline<F: JoltField>(polys: Vec<DensePolynomial<F>>) -> (F, f64) {
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Batch);
    let start = Instant::now();
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_p,_c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let elapsed = start.elapsed().as_secs_f64()*1000.0;
    (sumcheck.input_claim, elapsed)
}

fn timed_streaming_with_polys<F: JoltField>(polys: Vec<DensePolynomial<F>>) -> (F, f64) {
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Streaming);
    let start = Instant::now();
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_p,_c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let elapsed = start.elapsed().as_secs_f64()*1000.0;
    (sumcheck.input_claim, elapsed)
}

fn verify_latest<F: JoltField>(t: u32, d: u32) -> Result<(), Box<dyn std::error::Error>> {
    let polys = build_random_dense_polys::<F>(t, d, "fun");
    // Baseline
    let mut base = ProductSumcheck::from_polynomials_mode(polys.clone(), ExecutionMode::Batch);
    let mut t1 = Blake2bTranscript::new(b"sumcheck_experiment");
    let (p1,_c1) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut base, None, &mut t1);
    let mut vt1 = Blake2bTranscript::new(b"sumcheck_experiment");
    let acc1 = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));
    SingleSumcheck::verify::<F, Blake2bTranscript>(&base, &p1, Some(acc1), &mut vt1)?;
    // Streaming
    let mut sliced = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Streaming);
    let mut t2 = Blake2bTranscript::new(b"sumcheck_experiment");
    let (p2,_c2) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sliced, None, &mut t2);
    let mut vt2 = Blake2bTranscript::new(b"sumcheck_experiment");
    let acc2 = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));
    SingleSumcheck::verify::<F, Blake2bTranscript>(&sliced, &p2, Some(acc2), &mut vt2)?;
    Ok(())
}
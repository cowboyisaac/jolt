use clap::{Parser, Subcommand};
use ark_bn254::Fr;
use std::time::Instant;
use rayon::prelude::*;

// plotting moved to plotting.rs

mod product_sumcheck;
mod bench {
    pub mod plotting;
}
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::subprotocols::sumcheck::SingleSumcheck;
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
        
        /// Mode to run (0=Batch, 1=Tiling)
        #[arg(short, long, default_value = "0")]
        mode: u32,

        /// Tile length (number of pairs per tile). If omitted, uses default heuristic.
        #[arg(long = "tile-len")]
        tile_len: Option<usize>,

        /// Number of Rayon threads to use (overrides RAYON_NUM_THREADS)
        #[arg(long = "threads")]
        threads: Option<usize>,
    },
    /// Compare different sumcheck implementations
    Compare {
        /// Size of the sumcheck (2^T)
        #[arg(short = 'T', long = "T", default_value = "10")]
        t: u32,
        
        /// Degree of the polynomial
        #[arg(short, long, default_value = "2")]
        d: u32,
        /// Tile length (number of pairs per tile). If omitted, uses default heuristic.
        #[arg(long = "tile-len")]
        tile_len: Option<usize>,

        /// Number of Rayon threads to use (overrides RAYON_NUM_THREADS)
        #[arg(long = "threads")]
        threads: Option<usize>,
    },
    /// Batch compare across grids of T, d
    Batch {
        /// Sizes (2^T). Comma-separated list, e.g. 15,20,24
        #[arg(long = "T", value_delimiter = ',', required = true)]
        t_list: Vec<u32>,
        /// Degrees d. Comma-separated list, e.g. 2,3,4
        #[arg(long = "d", value_delimiter = ',', required = true)]
        d_list: Vec<u32>,
        /// Output image path (PNG/JPG). Relative paths are written to the current directory.
        #[arg(long = "out", default_value = "./bench_results.png")]
        out_path: String,
        /// Comma-separated list of thread counts to test, e.g. 8,16,32. If omitted, uses current default.
        #[arg(long = "threads", value_delimiter = ',')]
        threads: Option<Vec<usize>>,
        /// Comma-separated list of tile lengths to test.
        #[arg(long = "tile-len", value_delimiter = ',')]
        tile_lens: Option<Vec<usize>>,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run { t, d, mode, tile_len, threads } => {
            if let Some(n) = threads { let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global(); }
            run_single_experiment(t, d, mode, tile_len)
        }
        Commands::Compare { t, d, tile_len, threads } => {
            if let Some(n) = threads { let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global(); }
            match compare_implementations(t, d, tile_len) {
                Ok(()) => println!("\nCompare finished successfully."),
                Err(e) => eprintln!("\nCompare failed: {}", e),
            }
        }
        Commands::Batch { t_list, d_list, out_path, threads, tile_lens } => {
            run_batch_experiments(t_list, d_list, out_path, threads, tile_lens);
        }
    }
}

// Threads are auto-managed by rayon now.

fn run_single_experiment(t: u32, d: u32, mode: u32, tile_len: Option<usize>) {
    println!("Running sumcheck experiment: T={}, d={}, mode={}", t, d, mode);
    let total_start = Instant::now();
    let gen_start = Instant::now();
    let polys = build_random_dense_polys::<Fr>(t, d, "fun");
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    let (claim, total_proving_ms, boot_ms, recur_ms, input_ms, _pr_ms, _pr_len) = match mode {
        0 => timed_batch::<Fr>(polys),
        1 => timed_tiling_with_polys::<Fr>(polys, tile_len),
        _ => {
            println!("Unknown mode {}, falling back to mode 0", mode);
            timed_batch::<Fr>(build_random_dense_polys::<Fr>(t, d, "fun"))
        }
    };
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let threads = rayon::current_num_threads();
    let total_elems = (1u128 << t) * (d as u128);
    let per_thread_speed = total_elems as f64 / threads as f64;
    let throughput = if total_proving_ms > 0.0 {
        per_thread_speed / (total_proving_ms / 1000.0)
    } else {
        0.0
    };
    println!(
        "Completed: total={:.2}ms, gen={:.2}ms, input_claim={:.2}ms, boot-kernel={:.2}ms, recursive-kernel={:.2}ms, total_proving={:.2}ms, threads={}, per_thread_speed={:.0} elems, throughput_per_thread={:.2} elems/s, output={}",
        total_ms, gen_ms, input_ms, boot_ms, recur_ms, total_proving_ms, threads, per_thread_speed, throughput, claim
    );
}

fn compare_implementations(t: u32, d: u32, tile_len: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing sumcheck implementations: T={}, d={}", t, d);
    let overall_start = Instant::now();
    let gen_start = Instant::now();
    let polys = build_random_dense_polys::<Fr>(t, d, "fun");
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

    let clone_start = Instant::now();
    let polys_clone = polys.clone();
    let clone_ms = clone_start.elapsed().as_secs_f64() * 1000.0;

    let (claim_batch, t_batch, boot_batch_ms, recur_batch_ms, input_batch_ms, per_round_batch_ms, per_round_batch_len) = timed_batch::<Fr>(polys_clone);
    let (claim_tiling, t_tiling, boot_tiling_ms, recur_tiling_ms, input_tiling_ms, per_round_tiling_ms, _per_round_tiling_len) = timed_tiling_with_polys::<Fr>(polys, tile_len);

    let overall_ms = overall_start.elapsed().as_secs_f64() * 1000.0;
    let accounted = gen_ms + clone_ms + t_batch + t_tiling;
    let overhead_ms = (overall_ms - accounted).max(0.0);

    let threads = rayon::current_num_threads();
    let total_elems = (1u128 << t) * (d as u128);

    println!("Threads Used: {}", threads);
    println!("Results:");
    println!(
        "  Batch:  input-eval={:.2}ms | boot-kernel={:.2}ms | recursive-kernel={:.2}ms | claim={} | equal={}",
        input_batch_ms, boot_batch_ms, recur_batch_ms, claim_batch, claim_batch == claim_tiling
    );
    println!(
        "  Tiling{}: input-eval={:.2}ms | boot-kernel={:.2}ms | recursive-kernel={:.2}ms",
        tile_len.map(|v| format!(" (tile_len={})", v)).unwrap_or_default(),
        input_tiling_ms, boot_tiling_ms, recur_tiling_ms,
    );
    let sp_input = if input_tiling_ms > 0.0 { input_batch_ms / input_tiling_ms } else { 0.0 };
    let sp_boot = if boot_tiling_ms > 0.0 { boot_batch_ms / boot_tiling_ms } else { 0.0 };
    let sp_recur = if recur_tiling_ms > 0.0 { recur_batch_ms / recur_tiling_ms } else { 0.0 };
    println!("  Speedup (batch/tiling): input={:.2}x | boot={:.2}x | recur={:.2}x", sp_input, sp_boot, sp_recur);
    println!(
        "  Time breakdown: gen={:.2}ms | clone={:.2}ms | overhead={:.2}ms",
        gen_ms, clone_ms, overhead_ms
    );
    let total_batch_ms = boot_batch_ms + recur_batch_ms;
    let total_tiling_ms = boot_tiling_ms + recur_tiling_ms;
    let throughput_batch = if total_batch_ms > 0.0 {
        total_elems as f64 / threads as f64 / (total_batch_ms / 1000.0)
    } else {
        0.0
    };
    let throughput_tiling = if total_tiling_ms > 0.0 {
        total_elems as f64 / threads as f64 / (total_tiling_ms / 1000.0)
    } else {
        0.0
    };
    println!(
        "  Throughput per thread: batch={:.2} elems/s | tiling={:.2} elems/s",
        throughput_batch, throughput_tiling
    );
    let speedup = if t_tiling > 0.0 { t_batch / t_tiling } else { 0.0 };
    println!("  Speedup (Batch/Tiling): {:.2}x", speedup);
    // Per-round breakdown printed only in compare
    let rounds = std::cmp::min(per_round_batch_ms.len(), per_round_tiling_ms.len());
    if rounds > 0 {
        println!("\nPer-round timings and speedup (by vector length before binding):");
        println!("  {:>5} | {:>4} | {:>14} | {:>12} | {:>12} | {:>8}", "round", "t", "vec_len", "batch(ms)", "tiling(ms)", "speedup");
        for r in 1..rounds {
            // Length is identical across modes; use batch's recorded vector length
            let vec_len = *per_round_batch_len.get(r).unwrap_or(&0);
            let b = per_round_batch_ms[r];
            let tl = per_round_tiling_ms[r];
            let sp = if tl > 0.0 { b / tl } else { 0.0 };
            let t_round = if vec_len > 0 { (usize::BITS as usize - 1) - (vec_len.leading_zeros() as usize) } else { 0 };
            println!("  {:>5} | {:>4} | {:>14} | {:>12.2} | {:>12.2} | {:>7.2}x", r, t_round, vec_len, b, tl, sp);
        }
    }
    Ok(())
}

fn run_batch_experiments(t_list: Vec<u32>, d_list: Vec<u32>, out_path: String, threads: Option<Vec<usize>>, tile_lens: Option<Vec<usize>>) {
    let thread_variants: Vec<usize> = if let Some(v) = threads { if v.is_empty() { vec![rayon::current_num_threads()] } else { v } } else { vec![rayon::current_num_threads()] };
    let tile_len_variants: Vec<usize> = if let Some(v) = tile_lens { if v.is_empty() { vec![0usize] } else { v } } else { vec![0usize] };
    println!("Batch experiments: Ts={:?}, ds={:?}, threads={:?}, tile_lens={:?}", t_list, d_list, thread_variants, tile_len_variants);
    let mut rows: Vec<(u32, u32, usize, usize, f64, f64, f64, f64, usize, f64, f64, f64, f64, f64, f64)> = Vec::new();
    // (vec_len, batch_ms, tiling_ms, T, d, threads, tile_len)
    let mut repr_rounds: Option<(Vec<usize>, Vec<f64>, Vec<f64>, u32, u32, usize, usize)> = None;
    for &thr in &thread_variants {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(thr).build().expect("build thread pool");
        pool.install(|| {
            for &t in &t_list {
                for &d in &d_list {
                    let gen_start = Instant::now();
                    let polys_base = build_random_dense_polys::<Fr>(t, d, "fun");
                    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
                    // Run batch once per (t,d,threads)
                    let batch_clone = polys_base.clone();
                    let (claim_batch, t_batch, boot_batch_ms, recur_batch_ms, input_batch_ms, pr_ms_b, pr_len_b) = timed_batch::<Fr>(batch_clone);

                    for &tile_len in &tile_len_variants {
                        let overall_start = Instant::now();
                        let tile_len_opt = if tile_len == 0 { None } else { Some(tile_len) };
                        // Clone per tile_len for tiling runs
                        let polys_for_tiling = polys_base.clone();
                        let (claim_tiling, t_tiling, boot_tiling_ms, recur_tiling_ms, input_tiling_ms, pr_ms_t, _pr_len_t) = timed_tiling_with_polys::<Fr>(polys_for_tiling, tile_len_opt);
                        let overall_ms = overall_start.elapsed().as_secs_f64() * 1000.0;
                        let threads_here = rayon::current_num_threads();
                        let tile_len_val = tile_len_opt.unwrap_or(0);
                        println!(
                        "T={}, d={}, threads={}, tile_len={}\n  Batch:  input-claim={:.2}ms | boot-kernel={:.2}ms | recursive-kernel={:.2}ms | equal={}\n  Tiling: input-claim={:.2}ms | boot-kernel={:.2}ms | recursive-kernel={:.2}ms",
                        t, d, threads_here, tile_len_val,
                        input_batch_ms, boot_batch_ms, recur_batch_ms, claim_batch == claim_tiling,
                        input_tiling_ms, boot_tiling_ms, recur_tiling_ms
                        );
                        // Per-phase speedups (batch / tiling)
                        let sp_input = if input_tiling_ms > 0.0 { input_batch_ms / input_tiling_ms } else { 0.0 };
                        let sp_boot = if boot_tiling_ms > 0.0 { boot_batch_ms / boot_tiling_ms } else { 0.0 };
                        let sp_recur = if recur_tiling_ms > 0.0 { recur_batch_ms / recur_tiling_ms } else { 0.0 };
                        println!(
                            "  Speedup (batch/tiling): input={:.2}x | boot={:.2}x | recur={:.2}x",
                            sp_input, sp_boot, sp_recur
                        );
                        rows.push((t, d, threads_here, tile_len_opt.unwrap_or(0), gen_ms, t_batch, t_tiling, overall_ms, threads_here, boot_batch_ms, recur_batch_ms, boot_tiling_ms, recur_tiling_ms, input_batch_ms, input_tiling_ms));
                        // Save representative per-round data for the largest T encountered
                        let should_set = match &repr_rounds {
                            None => true,
                            Some((_l, _b, _ti, t_prev, _dprev, _thprev, _tlprev)) => t > *t_prev,
                        };
                        if should_set {
                            let mut xs: Vec<usize> = Vec::new();
                            let mut ys_b: Vec<f64> = Vec::new();
                            let mut ys_t: Vec<f64> = Vec::new();
                            let rounds = std::cmp::min(pr_ms_b.len(), pr_ms_t.len());
                            for r in 1..rounds {
                                xs.push(*pr_len_b.get(r).unwrap_or(&0));
                                ys_b.push(pr_ms_b[r]);
                                ys_t.push(pr_ms_t[r]);
                            }
                            repr_rounds = Some((xs, ys_b, ys_t, t, d, threads_here, tile_len_val));
                        }
                    }
                }
            }
        });
    }

    // Draw charts using the plotting module
    bench::plotting::draw_all_charts(&rows, &t_list, &d_list, &thread_variants, &out_path);
    if let Some((xs, ys_b, ys_t, t, d, thr, tile_len)) = repr_rounds {
        bench::plotting::draw_rounds_chart(&xs, &ys_b, &ys_t, t, d, thr, tile_len, &out_path);
        bench::plotting::draw_rounds_speedup_chart(&xs, &ys_b, &ys_t, &out_path);
        bench::plotting::draw_rounds_normalized_chart(&xs, &ys_b, &ys_t, &out_path);
        bench::plotting::draw_rounds_tail_chart(&xs, &ys_b, &ys_t, 8, &out_path);
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

    let n = 1usize.checked_shl(t).unwrap_or(0);
    let degree = d as usize;
    let base_seed = seed.as_bytes().iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
    // Small coefficient range to reduce conversion overhead and improve cache behavior.
    // Precompute a tiny lookup table once per call.
    let coeff_table: Vec<F> = (0u64..=16u64).map(|v| F::from_u64(v)).collect();

    (0..degree)
        .into_par_iter()
        .map(|poly_idx| {
            let mut coeffs = vec![F::zero(); n];
            let poly_seed = base_seed.wrapping_add(poly_idx as u64);
            coeffs
                .par_chunks_mut(1024)
                .enumerate()
                .for_each(|(chunk_i, chunk)| {
                    let start = chunk_i * 1024;
                    for (off, c) in chunk.iter_mut().enumerate() {
                        let i = start + off;
                        let r = mix64(poly_seed ^ (i as u64).wrapping_mul(0x9E3779B185EBCA87));
                        let val = 1 + ((r & 0xF) as u64);
                        // Clamp index for very strange situations, but val should always be between 1 and 16.
                        let lut_idx = if (val as usize) < coeff_table.len() {
                            val as usize
                        } else {
                            coeff_table.len() - 1
                        };
                        *c = coeff_table[lut_idx];
                    }
                });
            DensePolynomial::new(coeffs)
        })
        .collect()
}

// helpers removed; covered by timed_* functions

// The timed_batch and timed_tiling_with_polys functions below
// had wrong return signatures in the original code (should be -> (F, f64, f64, f64))
// but only returned (F, f64).
// Presume they should return input_ms and compute_ms as f64
// Adjust the helpers accordingly so calling sites work.

fn timed_batch<F: JoltField>(polys: Vec<DensePolynomial<F>>) -> (F, f64, f64, f64, f64, Vec<f64>, Vec<usize>) {
    let start = Instant::now();
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Batch, None);

    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_p, _c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    (sumcheck.input_claim, total_ms, sumcheck.boot_kernel_ms, sumcheck.recursive_kernel_ms, sumcheck.input_claim_ms, sumcheck.per_round_ms, sumcheck.per_round_len)
}

fn timed_tiling_with_polys<F: JoltField>(polys: Vec<DensePolynomial<F>>, tile_len: Option<usize>) -> (F, f64, f64, f64, f64, Vec<f64>, Vec<usize>) {
    let start = Instant::now();
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Tiling, tile_len);

    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let (_p, _c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    (sumcheck.input_claim, total_ms, sumcheck.boot_kernel_ms, sumcheck.recursive_kernel_ms, sumcheck.input_claim_ms, sumcheck.per_round_ms, sumcheck.per_round_len)
}

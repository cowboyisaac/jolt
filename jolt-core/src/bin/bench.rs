use clap::{Parser, Subcommand};
use ark_bn254::Fr;
use std::time::Instant;
use rayon::prelude::*;
use plotters::prelude::*;

mod product_sumcheck;
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
    /// Batch compare across grids of T, d
    Batch {
        /// Sizes (2^T). Comma-separated list, e.g. 15,20,24
        #[arg(long = "T", value_delimiter = ',', required = true)]
        t_list: Vec<u32>,
        /// Degrees d. Comma-separated list, e.g. 2,3,4
        #[arg(long = "d", value_delimiter = ',', required = true)]
        d_list: Vec<u32>,
        /// L1 data cache size in kB for tile sizing (streaming). Default 32.
        #[arg(long = "l1-kb", default_value = "32")]
        l1_kb: usize,
        /// Output image path (PNG/JPG). Relative paths are written to the current directory.
        #[arg(long = "out", default_value = "./bench_results.png")]
        out_path: String,
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
        Commands::Batch { t_list, d_list, l1_kb, out_path } => {
            run_batch_experiments(t_list, d_list, l1_kb, out_path);
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
    let (claim, total_proving_ms, input_ms, compute_ms) = match mode {
        0 => timed_batch::<Fr>(polys, l1_kb),
        1 => timed_tiling_with_polys::<Fr>(polys, l1_kb),
        _ => {
            println!("Unknown mode {}, falling back to mode 0", mode);
            timed_batch::<Fr>(build_random_dense_polys::<Fr>(t, d, "fun"), l1_kb)
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
        "Completed: total={:.2}ms, gen={:.2}ms, first_sum={:.2}ms, prove={:.2}ms, total_proving={:.2}ms, threads={}, per_thread_speed={:.0} elems, throughput_per_thread={:.2} elems/s, output={}",
        total_ms, gen_ms, input_ms, compute_ms, input_ms + compute_ms, threads, per_thread_speed, throughput, claim
    );
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

    let (claim_batch, t_batch, first_sum_batch_ms, prove_batch_ms) = timed_batch::<Fr>(polys_clone, l1_kb);
    let (claim_tiling, t_tiling, first_sum_tiling_ms, prove_tiling_ms) = timed_tiling_with_polys::<Fr>(polys, l1_kb);

    let overall_ms = overall_start.elapsed().as_secs_f64() * 1000.0;
    let accounted = gen_ms + clone_ms + t_batch + t_tiling;
    let overhead_ms = (overall_ms - accounted).max(0.0);

    let threads = rayon::current_num_threads();
    let total_elems = (1u128 << t) * (d as u128);

    // Print Batch timings
    println!(
        "Batch: total={:.2}ms (first_sum={:.2}ms, prove={:.2}ms), claim={} (equal to tiling: {})",
        t_batch, first_sum_batch_ms, prove_batch_ms, claim_batch, claim_batch == claim_tiling
    );

    // Print Tiling timings on a separate line for clarity
    println!(
        "Tiling: total={:.2}ms (first_sum={:.2}ms, prove={:.2}ms)",
        t_tiling, first_sum_tiling_ms, prove_tiling_ms,
    );

    // Extra clarification for first_sum
    if first_sum_batch_ms == 0.0 && first_sum_tiling_ms == 0.0 {
        println!("Note: first_sum=0.00ms indicates that the sum of the polynomial(s) over the hypercube was either negligible (optimized away) or too fast to measure precisely for this size.");
    }

    println!("Threads Used: {}", threads);
    println!(
        "Total={:.2}ms (gen={:.2}ms, clone={:.2}ms, batch={:.2}ms, tiling={:.2}ms, overhead={:.2}ms)",
        overall_ms, gen_ms, clone_ms, t_batch, t_tiling, overhead_ms
    );
    let total_batch_ms = first_sum_batch_ms + prove_batch_ms;
    let total_tiling_ms = first_sum_tiling_ms + prove_tiling_ms;
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
        "Throughput per thread: batch={:.2} elems/s, tiling={:.2} elems/s",
        throughput_batch, throughput_tiling
    );
    let speedup = if t_tiling > 0.0 { t_batch / t_tiling } else { 0.0 };
    println!("Speedup (Batch/Tiling): {:.2}x", speedup);
    Ok(())
}

fn run_batch_experiments(t_list: Vec<u32>, d_list: Vec<u32>, l1_kb: usize, out_path: String) {
    println!("Batch experiments: Ts={:?}, ds={:?}", t_list, d_list);
    let mut rows: Vec<(u32, u32, f64, f64, f64, f64, usize, f64, f64, f64, f64)> = Vec::new();
    for &t in &t_list {
        for &d in &d_list {
            let overall_start = Instant::now();
            let gen_start = Instant::now();
            let polys = build_random_dense_polys::<Fr>(t, d, "fun");
            let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
            let clone_start = Instant::now();
            let polys_clone = polys.clone();
            let _clone_ms = clone_start.elapsed().as_secs_f64() * 1000.0;
            let (claim_batch, t_batch, first_sum_batch_ms, prove_batch_ms) = timed_batch::<Fr>(polys_clone, l1_kb);
            let (claim_tiling, t_tiling, first_sum_tiling_ms, prove_tiling_ms) = timed_tiling_with_polys::<Fr>(polys, l1_kb);
            let overall_ms = overall_start.elapsed().as_secs_f64() * 1000.0;
            let threads = rayon::current_num_threads();
            let total_proving_batch = first_sum_batch_ms + prove_batch_ms;
            let total_proving_tiling = first_sum_tiling_ms + prove_tiling_ms;
            let speedup = if total_proving_tiling > 0.0 {
                total_proving_batch / total_proving_tiling
            } else {
                0.0
            };
            println!(
                "T={}, d={}, batch={:.2}ms, tiling={:.2}ms, total={:.2}ms, threads={}, equal={}",
                t, d, t_batch, t_tiling, overall_ms, threads, claim_batch == claim_tiling
            );
            println!(
                "   input: batch={:.2}ms, tiling={:.2}ms | prove: batch={:.2}ms, tiling={:.2}ms | total_proving speedup={:.2}x",
                first_sum_batch_ms, first_sum_tiling_ms, prove_batch_ms, prove_tiling_ms, speedup
            );
            rows.push((t, d, gen_ms, t_batch, t_tiling, overall_ms, threads, first_sum_batch_ms, prove_batch_ms, first_sum_tiling_ms, prove_tiling_ms));
        }
    }

    // Render a simple chart: x-axis T, two lines (batch, tiling) for each d; encode threads in legend
    let root = BitMapBackend::new(&out_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let y_max = rows.iter()
        .map(|r| (r.7 + r.8).max(r.9 + r.10))
        .fold(0.0, f64::max) * 1.2;
    let t_min = *t_list.iter().min().unwrap_or(&0) as i32;
    let t_max = *t_list.iter().max().unwrap_or(&0) as i32;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sumcheck Proving Time (ms) — L1={}kB, threads={}", l1_kb, rayon::current_num_threads()),
            ("sans-serif", 24)
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (t_min - 1)..(t_max + 1),
            0f64..y_max
        )
        .unwrap();

    chart.configure_mesh().x_desc("T").y_desc("ms").draw().unwrap();

    for &d in &d_list {
        let mut series_batch: Vec<(i32, f64)> = Vec::new();
        let mut series_tiling: Vec<(i32, f64)> = Vec::new();
        for &(t, dd, _gen, _pb, _pt, _tot, _thr, ib, cb, it, ct) in rows.iter() {
            if dd == d {
                series_batch.push((t as i32, ib + cb));
                series_tiling.push((t as i32, it + ct));
            }
        }
        chart
            .draw_series(LineSeries::new(series_batch, &RED))
            .unwrap()
            .label(format!("batch d={}", d))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .draw_series(LineSeries::new(series_tiling, &BLUE))
            .unwrap()
            .label(format!("tiling d={}", d))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    }
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Wrote chart: {}", out_path);
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

fn timed_batch<F: JoltField>(polys: Vec<DensePolynomial<F>>, l1_kb: usize) -> (F, f64, f64, f64) {
    let start = Instant::now();
    // In batch mode, input_claim is computed with a parallel map-reduce over i (as in compare mode).
    // Here we time the input_claim computation to match Compare mode for consistent reporting.
    let compute_start = Instant::now();
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Batch, Some(l1_kb));
    let first_sum = sumcheck.input_claim;
    let input_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let prove_start = Instant::now();
    let (_p, _c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    (first_sum, total_ms, input_ms, prove_ms)
}

fn timed_tiling_with_polys<F: JoltField>(polys: Vec<DensePolynomial<F>>, l1_kb: usize) -> (F, f64, f64, f64) {
    let start = Instant::now();
    let constructor_start = Instant::now();
    let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, ExecutionMode::Tiling, Some(l1_kb));
    let first_sum_ms = constructor_start.elapsed().as_secs_f64() * 1000.0; // first_sum time = constructor time

    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let first_sum = sumcheck.input_claim;
    let prove_start = Instant::now();
    let (_p, _c) = SingleSumcheck::prove::<F, Blake2bTranscript>(&mut sumcheck, None, &mut transcript);
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    (first_sum, total_ms, first_sum_ms, prove_ms)
}

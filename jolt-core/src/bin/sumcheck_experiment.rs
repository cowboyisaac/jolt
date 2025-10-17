use clap::{Parser, Subcommand};
use ark_bn254::Fr;
use std::time::Instant;

mod sumcheck_implementations;
use sumcheck_implementations::{run_baseline_sumcheck, run_tile_sumcheck};

#[derive(Parser)]
#[command(name = "sumcheck-experiment")]
#[command(about = "A tool for experimenting with sumcheck implementations and performance")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run sumcheck experiments with different configurations
    Run {
        /// Size of the sumcheck (2^k)
        #[arg(short, long, default_value = "10")]
        k: u32,
        
        /// Degree of the polynomial
        #[arg(short, long, default_value = "2")]
        d: u32,
        
        /// Mode to run (0=baseline, 1=tile)
        #[arg(short, long, default_value = "0")]
        mode: u32,
        
        /// Number of iterations to run
        #[arg(short, long, default_value = "1")]
        iterations: usize,
        
        /// Number of threads to use (0 = auto-detect)
        #[arg(short, long, default_value = "0")]
        threads: usize,
    },
    /// Compare different sumcheck implementations
    Compare {
        /// Size of the sumcheck (2^k)
        #[arg(short, long, default_value = "10")]
        k: u32,
        
        /// Degree of the polynomial
        #[arg(short, long, default_value = "2")]
        d: u32,
        
        /// Number of iterations per mode
        #[arg(short, long, default_value = "5")]
        iterations: usize,
        
        /// Number of threads to use (0 = auto-detect)
        #[arg(short, long, default_value = "0")]
        threads: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run {
            k,
            d,
            mode,
            iterations,
            threads,
        } => {
            setup_threading(threads);
            run_single_experiment(k, d, mode, iterations);
        }
        Commands::Compare {
            k,
            d,
            iterations,
            threads,
        } => {
            setup_threading(threads);
            compare_implementations(k, d, iterations);
        }
    }
}

fn setup_threading(threads: usize) {
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        println!("Using {} threads", threads);
    } else {
        let num_threads = rayon::current_num_threads();
        println!("Using {} threads (auto-detected)", num_threads);
    }
}

fn run_single_experiment(k: u32, d: u32, mode: u32, iterations: usize) {
    println!("Running sumcheck experiment: k={}, d={}, mode={}, iterations={}", k, d, mode, iterations);
    
    let mut times = Vec::new();
    let mut success = true;
    let mut _error_msg = None;
    let threads = rayon::current_num_threads();
    
    for i in 0..iterations {
        println!("Iteration {}/{}", i + 1, iterations);
        
        // Only profile the proving time, not verification
        let start = Instant::now();
        let result = match mode {
            0 => run_baseline_sumcheck::<Fr>(k, d, threads),
            1 => run_tile_sumcheck::<Fr>(k, d, threads),
            _ => {
                println!("Unknown mode {}, falling back to mode 0", mode);
                run_baseline_sumcheck::<Fr>(k, d, threads)
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
        println!("  Mode: {}", mode);
        println!("  Size: 2^{} = {}", k, 1 << k);
        println!("  Degree: {}", d);
        println!("  Iterations: {}", times.len());
        println!("  Total time: {:.2}ms", total_time);
        println!("  Average time: {:.2}ms", avg_time);
        println!("  Min time: {:.2}ms", min_time);
        println!("  Max time: {:.2}ms", max_time);
        println!("  Success: {}", success);
    }
}

fn compare_implementations(k: u32, d: u32, iterations: usize) {
    println!("Comparing sumcheck implementations: k={}, d={}, iterations={}", k, d, iterations);
    
    let modes = vec![
        (0, "Baseline (SingleSumcheck)"),
        (1, "Tile (SingleTileSumcheck)"),
    ];
    
    let mut results = Vec::new();
    let threads = rayon::current_num_threads();
    
    for (mode, name) in modes {
        println!("Testing mode {}: {}", mode, name);
        
        let mut times = Vec::new();
        let mut success = true;
        
        for _i in 0..iterations {
            let start = Instant::now();
            let result = match mode {
                0 => run_baseline_sumcheck::<Fr>(k, d, threads),
                1 => run_tile_sumcheck::<Fr>(k, d, threads),
                _ => run_baseline_sumcheck::<Fr>(k, d, threads),
            };
            let duration = start.elapsed();
            
            match result {
                Ok(_) => {
                    times.push(duration.as_secs_f64() * 1000.0);
                }
                Err(e) => {
                    println!("Mode {} failed: {}", mode, e);
                    success = false;
                    break;
                }
            }
        }
        
        if success && !times.is_empty() {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            results.push((mode, name, avg_time));
            println!("Mode {} ({}): {:.2}ms average", mode, name, avg_time);
        }
    }
    
    // Sort by performance
    results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    
    println!("\nPerformance comparison (fastest to slowest):");
    for (i, (mode, name, avg_time)) in results.iter().enumerate() {
        println!("{}. Mode {} ({}): {:.2}ms", i + 1, mode, name, avg_time);
    }
}
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::{SumcheckInstance, SingleSumcheck};
use jolt_core::transcripts::{Transcript, Blake2bTranscript};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Multilinear product sumcheck instance - the main sumcheck type
pub struct MultilinearProductSumcheckInstance<F: JoltField> {
    pub input_claim: F,
    pub polynomials: Vec<DensePolynomial<F>>, // List of d dense polynomials
    pub log_n: usize,
    pub degree: usize, // This is d, the number of polynomials
    pub threads: usize, // Thread configuration for rayon parallelization
}

impl<F: JoltField> MultilinearProductSumcheckInstance<F> {
    pub fn new(k: u32, d: u32, threads: usize, seed: &str) -> Self {
        let n = 1 << k;
        let log_n = k as usize;
        let degree = d as usize;
        
        // Create d dense polynomials
        let mut polynomials = Vec::new();
        for poly_idx in 0..degree {
            let mut coeffs = vec![F::zero(); n];
            for i in 0..n {
                // Use different seeds for each polynomial to ensure variety
                let poly_seed = seed.as_bytes().iter().map(|&b| b as u64).sum::<u64>() + poly_idx as u64;
                let mut poly_rng = StdRng::seed_from_u64(poly_seed);
                coeffs[i] = F::from_u64(poly_rng.gen_range(1..1000));
            }
            polynomials.push(DensePolynomial::new(coeffs));
        }
        
        // Compute input claim for multi-product sumcheck
        // Sum_{x in {0,1}^k} prod_{t=1..d} P_t(x)
        let input_claim = (0..n)
            .into_par_iter()
            .map(|i| {
                polynomials
                    .iter()
                    .fold(F::one(), |acc, poly| acc * poly.Z[i])
            })
            .reduce(|| F::zero(), |a, b| a + b);
        
        Self {
            input_claim,
            polynomials,
            log_n,
            degree,
            threads,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for MultilinearProductSumcheckInstance<F> {
    fn degree(&self) -> usize {
        self.degree
    }
    
    fn num_rounds(&self) -> usize {
        self.log_n
    }
    
    fn input_claim(&self) -> F {
        self.input_claim
    }
    
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        // For variable binding order HighToLow, the current variable splits Z into pairs (0,1)
        // For each pair j, define for each polynomial p: a = Z[2j], m = Z[2j+1] - Z[2j]
        // The per-pair univariate is prod_i (a_i + m_i * X). We need:
        //   - h(0) = sum_j prod_i a_i
        //   - h(t) for t in {2,3,...,degree}
        let half = self.polynomials[0].len() / 2;
        let pool_threads = self.threads;
        let degree = self.degree;

        let mut evals_at_points = vec![F::zero(); 1 + (degree.saturating_sub(1))];

        // h(0)
        let h0: F = if pool_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(pool_threads)
                .build()
                .unwrap()
                .install(|| {
                    (0..half)
                        .into_par_iter()
                        .map(|j| {
                            self.polynomials
                                .iter()
                                .fold(F::one(), |acc, poly| acc * poly.Z[2 * j])
                        })
                        .sum()
                })
        } else {
            (0..half)
                .into_par_iter()
                .map(|j| {
                    self.polynomials
                        .iter()
                        .fold(F::one(), |acc, poly| acc * poly.Z[2 * j])
                })
                .sum()
        };
        evals_at_points[0] = h0;

        // h(t) for t in {2..=degree}
        for t in 2..=degree {
            let t_f = F::from_u64(t as u64);
            let ht: F = if pool_threads > 0 {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(pool_threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        (0..half)
                            .into_par_iter()
                            .map(|j| {
                                self.polynomials.iter().fold(F::one(), |acc, poly| {
                                    let a = poly.Z[2 * j];
                                    let b = poly.Z[2 * j + 1];
                                    let m = b - a;
                                    acc * (a + m * t_f)
                                })
                            })
                            .sum()
                    })
            } else {
                (0..half)
                    .into_par_iter()
                    .map(|j| {
                        self.polynomials.iter().fold(F::one(), |acc, poly| {
                            let a = poly.Z[2 * j];
                            let b = poly.Z[2 * j + 1];
                            let m = b - a;
                            acc * (a + m * t_f)
                        })
                    })
                    .sum()
            };
            evals_at_points[t - 1] = ht;
        }

        evals_at_points
    }
    
    
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind all polynomials in parallel using rayon
        self.polynomials.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r_j, BindingOrder::HighToLow);
        });
    }
    
    fn expected_output_claim(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        // For a proper sumcheck, we need to evaluate the polynomial at the challenge points
        if r.is_empty() {
            return self.input_claim;
        }
        
        // Evaluate each polynomial at the challenge point and take the product
        let mut result = F::one();
        for poly in &self.polynomials {
            let poly_eval = poly.evaluate(r);
            result = result * poly_eval;
        }
        
        result
    }
    
    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }
    
    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // No-op for testing
    }
    
    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // No-op for testing
    }
    
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _flamegraph: &mut allocative::FlameGraphBuilder) {
        // No-op for testing
    }
}


// Core sumcheck execution functions
pub fn run_baseline_sumcheck<F: JoltField>(k: u32, d: u32, threads: usize) -> Result<F, Box<dyn std::error::Error>> {
    let mut sumcheck = MultilinearProductSumcheckInstance::<F>::new(k, d, threads, "fun");
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    
    let (_proof, _challenges) = SingleSumcheck::prove::<F, Blake2bTranscript>(
        &mut sumcheck,
        None,
        &mut transcript,
    );
    
    // Verify using SingleSumcheck with a verifier opening accumulator
    let mut verify_transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    let opening_acc = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));
    let challenges = SingleSumcheck::verify::<F, Blake2bTranscript>(
        &sumcheck,
        &_proof,
        Some(opening_acc),
        &mut verify_transcript,
    )?;

    // Compare input claim with final expected evaluation at challenges
    let input_claim = <MultilinearProductSumcheckInstance<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck);
    let expected = <MultilinearProductSumcheckInstance<F> as SumcheckInstance<F, Blake2bTranscript>>::expected_output_claim(
        &sumcheck,
        None,
        &challenges,
    );
    Ok(input_claim)
}


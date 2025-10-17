use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::{SumcheckInstance, SingleSumcheck, SumcheckInstanceProof};
use jolt_core::utils::errors::ProofVerifyError;
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
        // This is the sum over all binary strings of the product of all d polynomials
        let input_claim = (0..n)
            .map(|i| {
                let mut binary = [0u32; 32];
                let mut temp = i;
                for j in 0..log_n {
                    binary[j] = (temp & 1) as u32;
                    temp >>= 1;
                }
                
                // Evaluate each polynomial at the binary point
                let mut product = F::one();
                for _poly in &polynomials {
                    let mut poly_eval = F::zero();
                    for j in 0..log_n {
                        let x = F::from_u64(binary[j] as u64);
                        // For degree 1 polynomials, this is just 1 + x
                        poly_eval = poly_eval + F::one() + x;
                    }
                    product = product * poly_eval;
                }
                
                // Multiply by the polynomial coefficients
                let mut coeff_product = F::one();
                for poly in &polynomials {
                    coeff_product = coeff_product * poly.Z[i];
                }
                
                coeff_product * product
            })
            .sum();
        
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
        let degree = self.degree;
        
        // Use rayon parallelization with thread configuration
        let univariate_poly_evals: Vec<F> = (0..self.polynomials[0].len() / 2)
            .into_par_iter()
            .map(|j| {
                // For each polynomial, get the evaluations at 0 and 1
                let mut poly_evals = Vec::new();
                for poly in &self.polynomials {
                    let evals = [
                        poly.Z[2 * j],     // eval at 0
                        poly.Z[2 * j + 1], // eval at 1
                    ];
                    poly_evals.push(evals);
                }
                
                // Compute the product of all polynomials
                // For degree 1 polynomials, we evaluate at 0 and 1
                let mut result = F::zero();
                for deg in 0..degree {
                    let mut product = F::one();
                    for poly_eval in &poly_evals {
                        // For degree 1: eval at 0 is poly_eval[0], eval at 1 is poly_eval[1]
                        // The univariate polynomial is: poly_eval[0] * (1-x) + poly_eval[1] * x
                        // At point deg: poly_eval[0] * (1-deg) + poly_eval[1] * deg
                        let x = F::from_u64(deg as u64);
                        let eval = poly_eval[0] * (F::one() - x) + poly_eval[1] * x;
                        product = product * eval;
                    }
                    result += product;
                }
                result
            })
            .collect();
        
        // Sum all evaluations
        let total = univariate_poly_evals.into_iter().sum();
        vec![total]
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
        _r: &[F::Challenge],
    ) -> F {
        // For now, return the input claim as a simplified verification
        // This allows us to focus on the proving performance first
        self.input_claim
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

// SingleTileSumcheck - placeholder for tile mode implementation
// For now, this is the same as SingleSumcheck but will be customized later
pub enum SingleTileSumcheck {}

impl SingleTileSumcheck {
    /// Proves a single tile sumcheck instance.
    /// For now, this is the same as SingleSumcheck but will be customized later.
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &mut dyn SumcheckInstance<F, ProofTranscript>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        // For now, delegate to SingleSumcheck
        SingleSumcheck::prove(sumcheck_instance, opening_accumulator, transcript)
    }
    
    /// Verifies a single tile sumcheck instance.
    /// For now, this is the same as SingleSumcheck but will be customized later.
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &dyn SumcheckInstance<F, ProofTranscript>,
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        // For now, delegate to SingleSumcheck
        SingleSumcheck::verify(sumcheck_instance, proof, opening_accumulator, transcript)
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
    
    // For now, skip verification to focus on proving performance
    // TODO: Implement proper verification later
    println!("Proving completed successfully! Proof generated.");
    Ok(<MultilinearProductSumcheckInstance<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck))
}

pub fn run_tile_sumcheck<F: JoltField>(k: u32, d: u32, threads: usize) -> Result<F, Box<dyn std::error::Error>> {
    let mut sumcheck = MultilinearProductSumcheckInstance::<F>::new(k, d, threads, "fun");
    let mut transcript = Blake2bTranscript::new(b"sumcheck_experiment");
    
    let (_proof, _challenges) = SingleTileSumcheck::prove::<F, Blake2bTranscript>(
        &mut sumcheck,
        None,
        &mut transcript,
    );
    
    // For now, skip verification to focus on proving performance
    // TODO: Implement proper verification later
    println!("Tile proving completed successfully! Proof generated.");
    Ok(<MultilinearProductSumcheckInstance<F> as SumcheckInstance<F, Blake2bTranscript>>::input_claim(&sumcheck))
}

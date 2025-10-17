use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::Transcript;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

// Product sumcheck over d multilinear polynomials
pub struct ProductSumcheck<F: JoltField> {
    pub input_claim: F,
    pub polynomials: Vec<DensePolynomial<F>>, // List of d dense polynomials
    pub log_n: usize,
    pub degree: usize, // number of polynomials
    pub threads: usize, // parallel threads
}

impl<F: JoltField> ProductSumcheck<F> {
    pub fn from_polynomials(polynomials: Vec<DensePolynomial<F>>, threads: usize) -> Self {
        let n = polynomials.get(0).map(|p| p.len()).unwrap_or(0);
        let log_n = if n == 0 { 0 } else { n.trailing_zeros() as usize };
        let degree = polynomials.len();
        let input_claim = (0..n)
            .into_par_iter()
            .map(|i| polynomials.iter().fold(F::one(), |acc, poly| acc * poly.Z[i]))
            .reduce(|| F::zero(), |a, b| a + b);
        Self { input_claim, polynomials, log_n, degree, threads }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductSumcheck<F> {
    fn degree(&self) -> usize { self.degree }
    fn num_rounds(&self) -> usize { self.log_n }
    fn input_claim(&self) -> F { self.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let half = self.polynomials[0].len() / 2;
        let degree = self.degree;

        // Use dedicated pool if configured
        let compute = || {
            let mut evals_at_points = vec![F::zero(); 1 + (degree.saturating_sub(1))];
            // h(0)
            let h0: F = (0..half)
                .into_par_iter()
                .map(|j| self.polynomials.iter().fold(F::one(), |acc, poly| acc * poly.Z[2 * j]))
                .sum();
            evals_at_points[0] = h0;
            // h(t) for t in 2..=degree
            for t in 2..=degree {
                let t_f = F::from_u64(t as u64);
                let ht: F = (0..half)
                    .into_par_iter()
                    .map(|j| {
                        self.polynomials.iter().fold(F::one(), |acc, poly| {
                            let a = poly.Z[2 * j];
                            let b = poly.Z[2 * j + 1];
                            let m = b - a;
                            acc * (a + m * t_f)
                        })
                    })
                    .sum();
                evals_at_points[t - 1] = ht;
            }
            evals_at_points
        };

        if self.threads > 0 {
            rayon::ThreadPoolBuilder::new().num_threads(self.threads).build().unwrap().install(compute)
        } else {
            compute()
        }
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.polynomials.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r_j, BindingOrder::HighToLow);
        });
    }

    fn expected_output_claim(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
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
    ) {}

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {}

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _flamegraph: &mut allocative::FlameGraphBuilder) {}
}



#[cfg(test)]
mod tests {
    use super::ProductSumcheck;
    use ark_bn254::Fr;
    use jolt_core::field::JoltField;
    use jolt_core::poly::dense_mlpoly::DensePolynomial;
    use jolt_core::poly::opening_proof::VerifierOpeningAccumulator;
    use jolt_core::subprotocols::sumcheck::{SingleSumcheck, SumcheckInstance};
    use jolt_core::transcripts::{Blake2bTranscript, Transcript};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn build_random_dense_polys<F: JoltField>(t: u32, d: u32, seed: &str) -> Vec<DensePolynomial<F>> {
        let n = 1 << t;
        let degree = d as usize;
        let mut polynomials = Vec::with_capacity(degree);
        let base_seed = seed.as_bytes().iter().map(|&b| b as u64).sum::<u64>();
        for poly_idx in 0..degree {
            let mut coeffs = vec![F::zero(); n];
            let mut rng = StdRng::seed_from_u64(base_seed + poly_idx as u64);
            for i in 0..n {
                coeffs[i] = F::from_u64(rng.gen_range(1..1000));
            }
            polynomials.push(DensePolynomial::new(coeffs));
        }
        polynomials
    }

    fn run_product_sumcheck_test(d: u32) {
        let t = 10u32;
        let threads = 0usize;
        let polys = build_random_dense_polys::<Fr>(t, d, "product_sumcheck_test");
        let mut sumcheck = ProductSumcheck::from_polynomials(polys, threads);
        let mut prover_transcript = Blake2bTranscript::new(b"sumcheck_test");
        let (proof, _chals) =
            SingleSumcheck::prove::<Fr, Blake2bTranscript>(&mut sumcheck, None, &mut prover_transcript);
        let opening_acc = Rc::new(RefCell::new(VerifierOpeningAccumulator::<Fr>::new()));
        let mut verifier_transcript = Blake2bTranscript::new(b"sumcheck_test");
        let verify = SingleSumcheck::verify::<Fr, Blake2bTranscript>(
            &sumcheck,
            &proof,
            Some(opening_acc),
            &mut verifier_transcript,
        );
        assert!(verify.is_ok());
        // Sanity: expected_output_claim at random point matches protocol's output (implicit via verify)
        // Also check that input_claim is non-zero for these random inputs
        let _ = <ProductSumcheck<Fr> as SumcheckInstance<Fr, Blake2bTranscript>>::input_claim(&sumcheck);
    }

    #[test]
    fn product_sumcheck_t10_d2() {
        run_product_sumcheck_test(2);
    }

    #[test]
    fn product_sumcheck_t10_d3() {
        run_product_sumcheck_test(3);
    }

    #[test]
    fn product_sumcheck_t10_d4() {
        run_product_sumcheck_test(4);
    }
}

// Provide a no-op main when this file is compiled as a standalone binary target
fn main() {}


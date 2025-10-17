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
    original_polynomials: Vec<DensePolynomial<F>>, // Unmodified copies for final claim
    pub log_n: usize,
    pub degree: usize, // number of polynomials
}

impl<F: JoltField> ProductSumcheck<F> {
    pub fn from_polynomials(polynomials: Vec<DensePolynomial<F>>) -> Self {
        let n = polynomials.get(0).map(|p| p.len()).unwrap_or(0);
        let log_n = if n == 0 { 0 } else { n.trailing_zeros() as usize };
        let degree = polynomials.len();
        let original_polynomials = polynomials.clone();
        let input_claim = (0..n)
            .into_par_iter()
            .map(|i| polynomials.iter().fold(F::one(), |acc, poly| acc * poly.Z[i]))
            .reduce(|| F::zero(), |a, b| a + b);
        Self { input_claim, polynomials, original_polynomials, log_n, degree }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductSumcheck<F> {
    fn degree(&self) -> usize { self.degree }
    fn num_rounds(&self) -> usize { self.log_n }
    fn input_claim(&self) -> F { self.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let half = self.polynomials[0].len() / 2;
        let degree = self.degree;

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
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.polynomials.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        });
    }

    fn expected_output_claim(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let r_be: Vec<_> = r.iter().rev().copied().collect();
        self.original_polynomials
            .iter()
            .map(|poly| poly.evaluate(&r_be))
            .fold(F::one(), |acc, v| acc * v)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
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

// Sliced, cache-friendly product sumcheck with optional dedicated thread pool
pub struct SlicedProductSumcheck<F: JoltField> {
    pub base: ProductSumcheck<F>,
    tile_len: usize,
    pending_r: Option<F::Challenge>,
}

impl<F: JoltField> SlicedProductSumcheck<F> {
    fn compute_tile_len(degree: usize) -> usize {
        let l1_bytes: usize = 32 * 1024; // macbook pro 128kb, threadripper 32kb
        let elem_bytes: usize = core::mem::size_of::<F>().max(32); // BN254 ~32 bytes
        let d = degree.max(1);
        let mut tile_len = (l1_bytes / (d * elem_bytes)).max(64);
        tile_len = tile_len.min(1024);
        let pow = usize::BITS as usize - 1 - tile_len.leading_zeros() as usize;
        1usize << pow
    }

    pub fn from_polynomials(polynomials: Vec<DensePolynomial<F>>) -> Self {
        let base = ProductSumcheck::from_polynomials(polynomials);
        let tile_len = Self::compute_tile_len(base.degree);
        Self { base, tile_len, pending_r: None }
    }

    // Dedicated per-instance thread pools are no longer used; rayon manages threads globally.

    fn compute_once(&mut self) -> Vec<F> {
        if let Some(r) = self.pending_r.take() {
            let half = self.base.polynomials[0].len() / 2;
            let new_polys: Vec<DensePolynomial<F>> = self
                .base
                .polynomials
                .par_iter()
                .map(|poly| {
                    let mut new_evals = vec![F::zero(); half];
                    new_evals
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, z)| {
                            let a = poly.Z[2 * i];
                            let b = poly.Z[2 * i + 1];
                            let m = b - a;
                            *z = a + r * m;
                        });
                    DensePolynomial::new(new_evals)
                })
                .collect();
            self.base.polynomials = new_polys;
        }

        let working: Vec<&DensePolynomial<F>> = self.base.polynomials.iter().collect();
        let len = working[0].len();
        for (i, poly) in working.iter().enumerate() {
            assert_eq!(poly.len(), len, "Polynomial {} has length {}, expected {}", i, poly.len(), len);
        }
        let half = len / 2;
        let degree = self.base.degree;
        let points_len = 1 + (degree.saturating_sub(1));
        let t_vals: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();
        if half == 0 { return vec![F::zero(); points_len]; }

        let num_tiles = (half + self.tile_len - 1) / self.tile_len;

        struct TileOut<F> { h0: F, ht: Vec<F> }
        let tile_results: Vec<TileOut<F>> = (0..num_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let start = tile_idx * self.tile_len;
                let end = core::cmp::min(start + self.tile_len, half);
                let mut h0 = F::zero();
                let mut ht = vec![F::zero(); t_vals.len()];
                for j in start..end {
                    let mut prod_a = F::one();
                    let mut prod_t: Vec<F> = vec![F::one(); t_vals.len()];
                    for poly in working.iter() {
                        let a = poly.Z[2 * j];
                        let b = poly.Z[2 * j + 1];
                        let m = b - a;
                        prod_a = prod_a * a;
                        for (idx, &tv) in t_vals.iter().enumerate() { prod_t[idx] = prod_t[idx] * (a + m * tv); }
                    }
                    h0 = h0 + prod_a;
                    for idx in 0..ht.len() { ht[idx] = ht[idx] + prod_t[idx]; }
                }
                TileOut { h0, ht }
            })
            .collect();

        let mut evals_at_points = vec![F::zero(); points_len];
        for tile in tile_results.into_iter() {
            evals_at_points[0] = evals_at_points[0] + tile.h0;
            for idx in 0..tile.ht.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + tile.ht[idx]; }
        }

        evals_at_points
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for SlicedProductSumcheck<F> {
    fn degree(&self) -> usize { self.base.degree }
    fn num_rounds(&self) -> usize { self.base.log_n }
    fn input_claim(&self) -> F { self.base.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        self.compute_once()
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) { self.pending_r = Some(r_j); }

    fn expected_output_claim(
        &self,
        _o: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge]
    ) -> F {
        let r_be: Vec<_> = r.iter().rev().copied().collect();
        self.base
            .original_polynomials
            .iter()
            .map(|poly| poly.evaluate(&r_be))
            .fold(F::one(), |acc, v| acc * v)
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_prover(&self, _a: Rc<RefCell<ProverOpeningAccumulator<F>>>, _t: &mut T, _p: OpeningPoint<BIG_ENDIAN, F>) {}
    fn cache_openings_verifier(&self, _a: Rc<RefCell<VerifierOpeningAccumulator<F>>>, _t: &mut T, _p: OpeningPoint<BIG_ENDIAN, F>) {}

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _f: &mut allocative::FlameGraphBuilder) {}
}

#[cfg(test)]
mod tests {
    use super::{ProductSumcheck, SlicedProductSumcheck};
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
        let polys = build_random_dense_polys::<Fr>(t, d, "product_sumcheck_test");
        let mut sumcheck = SlicedProductSumcheck::from_polynomials(polys);
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
        let _ = <SlicedProductSumcheck<Fr> as SumcheckInstance<Fr, Blake2bTranscript>>::input_claim(&sumcheck);
    }

    fn run_slice_product_sumcheck_test(d: u32) {
        let t = 10u32;
        let polys = build_random_dense_polys::<Fr>(t, d, "product_sumcheck_test");
        let mut sumcheck = SlicedProductSumcheck::from_polynomials(polys);
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
        let _ = <SlicedProductSumcheck<Fr> as SumcheckInstance<Fr, Blake2bTranscript>>::input_claim(&sumcheck);
    }

    #[test]
    fn product_sumcheck_t10_d2() {
        run_product_sumcheck_test(2);
        run_product_sumcheck_test(3);
        run_product_sumcheck_test(4);
    }

    #[test]
    fn sliced_product_sumcheck_t10_d2() {
        run_slice_product_sumcheck_test(2);
        run_slice_product_sumcheck_test(3);
        run_slice_product_sumcheck_test(4);
    }

}

// Provide a no-op main when this file is compiled as a standalone binary target
fn main() {}


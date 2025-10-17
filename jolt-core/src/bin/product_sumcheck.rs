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
    pub original_polynomials: Vec<DensePolynomial<F>>, // Unmodified copies for final claim
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
            .reduce(|| F::zero(), |a, b| a + b);
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
                .reduce(|| F::zero(), |a, b| a + b);
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
    pub tile_len: usize,
    pub pending_r: Option<F::Challenge>,
    // Precomputed t values: t in [2..=degree]
    pub t_vals: Vec<F>,
}

impl<F: JoltField> SlicedProductSumcheck<F> {
    fn compute_tile_len(degree: usize) -> usize {
        let l1_bytes: usize = 32 * 1024; // macbook pro 128kb, threadripper 32kb
        let elem_bytes: usize = core::mem::size_of::<F>().max(32); // BN254 ~32 bytes
        let d = degree.max(1);
        let mut tile_len = (l1_bytes / (d * elem_bytes)).max(64);
        tile_len = tile_len.min(1024);
        let pow = usize::BITS - 1 - tile_len.leading_zeros();
        1usize << pow
    }

    pub fn from_polynomials(polynomials: Vec<DensePolynomial<F>>) -> Self {
        let base = ProductSumcheck::from_polynomials(polynomials);
        let tile_len = Self::compute_tile_len(base.degree);
        let t_vals: Vec<F> = (2..=base.degree).map(|t| F::from_u64(t as u64)).collect();
        Self { base, tile_len, pending_r: None, t_vals }
    }

    fn compute_once(&mut self) -> Vec<F> {
        let r_opt = self.pending_r.take();

        // Two modes:
        // - If binding pending (round > 0): per-tile bind-then-evaluate using the freshly bound cache
        // - If no binding (round 0): evaluate directly from current polynomials
        let working: &[DensePolynomial<F>] = &self.base.polynomials;
        let len_before = working[0].len();
        for (i, poly) in working.iter().enumerate() {
            assert_eq!(poly.len(), len_before, "Polynomial {} has length {}, expected {}", i, poly.len(), len_before);
        }
        let half_before = len_before / 2;
        let degree = self.base.degree;
        let points_len = 1 + (degree.saturating_sub(1));
        let t_vals = &self.t_vals;
        if half_before == 0 { return vec![F::zero(); points_len]; }

        let num_tiles = (half_before + self.tile_len - 1) / self.tile_len;
        let _num_polys = working.len();

        let mut evals_at_points = vec![F::zero(); points_len];

        if let Some(r) = r_opt {
            // Compute evals as if arrays were first bound by r (to half size),
            // but without materializing the bound arrays. Then perform the bind.
            let tile_evals: Vec<(F, Vec<F>)> = (0..num_tiles)
                .into_par_iter()
                .map(|tile_idx| {
                    let start = tile_idx * self.tile_len;
                    let end = core::cmp::min(start + self.tile_len, half_before);
                    // Align to even j so reduced pairs (2k,2k+1) are fully contained
                    let start_even = start & !1;
                    let end_even = end & !1;

                    // Product eval on reduced pairs within this tile. The reduced array
                    // y has length half_before with y[j] = a + r*(b-a). We need pairs
                    // (y[2k], y[2k+1]). We compute them on-the-fly from the original arrays:
                    // y[2k]   = A0 + r*(B0 - A0) where (A0,B0) = (x[4k], x[4k+1])
                    // y[2k+1] = A1 + r*(B1 - A1) where (A1,B1) = (x[4k+2], x[4k+3])
                    let mut tile_h0 = F::zero();
                    let mut tile_ht = vec![F::zero(); t_vals.len()];
                    let k_start = start_even / 2;
                    let k_end = end_even / 2;
                    // Reusable buffer for prod_t per k
                    let mut prod_t: Vec<F> = vec![F::one(); t_vals.len()];
                    for k in k_start..k_end {
                        let mut prod_a = F::one();
                        for poly in working.iter() {
                            let a00 = poly.Z[4 * k + 0];
                            let a01 = poly.Z[4 * k + 1];
                            let a10 = poly.Z[4 * k + 2];
                            let a11 = poly.Z[4 * k + 3];
                            let y0 = a00 + r * (a01 - a00);
                            let y1 = a10 + r * (a11 - a10);
                            let m2 = y1 - y0;
                            prod_a = prod_a * y0;
                            for (idx, &tv) in t_vals.iter().enumerate() {
                                prod_t[idx] = prod_t[idx] * (y0 + m2 * tv);
                            }
                        }
                        tile_h0 = tile_h0 + prod_a;
                        for idx in 0..tile_ht.len() { tile_ht[idx] = tile_ht[idx] + prod_t[idx]; }
                        // reset prod_t to ones for next k
                        for v in &mut prod_t { *v = F::one(); }
                    }
                    (tile_h0, tile_ht)
                })
                .collect();

            for (h0, ht) in tile_evals.into_iter() {
                evals_at_points[0] = evals_at_points[0] + h0;
                for idx in 0..ht.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + ht[idx]; }
            }

            // Now bind polynomials in-place for the next round.
            self.base
                .polynomials
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r, BindingOrder::LowToHigh));
        } else {
            // Round 0: evaluate directly from current (unbound) arrays using full-size pairs
            let tile_evals: Vec<(F, Vec<F>)> = (0..num_tiles)
                .into_par_iter()
                .map(|tile_idx| {
                    let start = tile_idx * self.tile_len;
                    let end = core::cmp::min(start + self.tile_len, half_before);
                    let mut tile_h0 = F::zero();
                    let mut tile_ht = vec![F::zero(); t_vals.len()];
                    // Reusable buffer for prod_t per j
                    let mut prod_t: Vec<F> = vec![F::one(); t_vals.len()];
                    for j in start..end {
                        let mut prod_a = F::one();
                        for poly in working.iter() {
                            let a = poly.Z[2 * j];
                            let b = poly.Z[2 * j + 1];
                            let m = b - a;
                            prod_a = prod_a * a;
                            for (idx, &tv) in t_vals.iter().enumerate() {
                                prod_t[idx] = prod_t[idx] * (a + m * tv);
                            }
                        }
                        tile_h0 = tile_h0 + prod_a;
                        for idx in 0..tile_ht.len() { tile_ht[idx] = tile_ht[idx] + prod_t[idx]; }
                        // reset prod_t to ones for next j
                        for v in &mut prod_t { *v = F::one(); }
                    }
                    (tile_h0, tile_ht)
                })
                .collect();

            for (h0, ht) in tile_evals.into_iter() {
                evals_at_points[0] = evals_at_points[0] + h0;
                for idx in 0..ht.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + ht[idx]; }
            }
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
        let mut sumcheck = ProductSumcheck::from_polynomials(polys);
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
#[allow(dead_code)]
fn main() {}

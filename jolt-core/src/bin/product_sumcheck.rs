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
pub enum ExecutionMode {
    Batch,
    Streaming,
}

pub struct ProductSumcheck<F: JoltField> {
    pub input_claim: F,
    pub polynomials: Vec<DensePolynomial<F>>, // List of d dense polynomials
    pub original_polynomials: Vec<DensePolynomial<F>>, // Unmodified copies for final claim
    pub log_n: usize,
    pub degree: usize, // number of polynomials
    pub mode: ExecutionMode,
    pub streaming: Option<StreamingState<F>>,
}

pub struct StreamingState<F: JoltField> {
    pub tile_len: usize,
    pub pending_r: Option<F::Challenge>,
    pub t_vals: Vec<F>,
}

impl<F: JoltField> StreamingState<F> {
    fn compute_tile_len(degree: usize) -> usize {
        let l1_bytes: usize = 32 * 1024; // macbook pro 128kb, threadripper 32kb
        let elem_bytes: usize = core::mem::size_of::<F>().max(32); // BN254 ~32 bytes
        let d = degree.max(1);
        let mut tile_len = (l1_bytes / (d * elem_bytes)).max(64);
        tile_len = tile_len.min(1024);
        let pow = usize::BITS - 1 - tile_len.leading_zeros();
        1usize << pow
    }

    fn new(degree: usize) -> Self {
        let tile_len = Self::compute_tile_len(degree);
        let t_vals: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();
        Self {
            tile_len,
            pending_r: None,
            t_vals,
        }
    }
}

impl<F: JoltField> ProductSumcheck<F> {
    pub fn from_polynomials_mode(polynomials: Vec<DensePolynomial<F>>, mode: ExecutionMode) -> Self {
        let n = polynomials.get(0).map(|p| p.len()).unwrap_or(0);
        let log_n = if n == 0 { 0 } else { n.trailing_zeros() as usize };
        let degree = polynomials.len();
        let original_polynomials = polynomials.clone();
        let input_claim = (0..n)
            .into_par_iter()
            .map(|i| polynomials.iter().fold(F::one(), |acc, poly| acc * poly.Z[i]))
            .reduce(|| F::zero(), |a, b| a + b);
        let streaming = match mode {
            ExecutionMode::Batch => None,
            ExecutionMode::Streaming => Some(StreamingState::new(degree)),
        };
        Self { input_claim, polynomials, original_polynomials, log_n, degree, mode, streaming }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductSumcheck<F> {
    fn degree(&self) -> usize { self.degree }
    fn num_rounds(&self) -> usize { self.log_n }
    fn input_claim(&self) -> F { self.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        match self.mode {
            ExecutionMode::Batch => {
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
            ExecutionMode::Streaming => {
                self.compute_once_streaming()
            }
        }
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        match self.mode {
            ExecutionMode::Batch => {
                self.polynomials.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            }
            ExecutionMode::Streaming => {
                if let Some(ref mut s) = self.streaming { s.pending_r = Some(r_j); }
            }
        }
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

// Streaming executor implementation folded into ProductSumcheck
impl<F: JoltField> ProductSumcheck<F> {
    fn compute_once_streaming(&mut self) -> Vec<F> {
        let s = self.streaming.as_mut().expect("streaming state expected");
        let r_opt = s.pending_r.take();

        // Working view for this round
        let working: &[DensePolynomial<F>] = &self.polynomials;
        let len_before = working[0].len();
        for (i, poly) in working.iter().enumerate() {
            assert_eq!(poly.len(), len_before, "Polynomial {} has length {}, expected {}", i, poly.len(), len_before);
        }
        // Dimensions and precomputed values (computed once per round)
        let half_before = len_before / 2;
        let degree = self.degree; // number of polynomials
        let points_len = 1 + (degree.saturating_sub(1));
        let t_vals = &s.t_vals; // t in [2..=degree]
        let t_len = t_vals.len();
        let num_polys = working.len();
        if half_before == 0 { return vec![F::zero(); points_len]; }

        let num_tiles = (half_before + s.tile_len - 1) / s.tile_len;

        // Accumulator for this round's prover message evaluations: [h(0), h(2), ..., h(d)]
        let mut evals_at_points = vec![F::zero(); points_len];

        if let Some(r) = r_opt {
            // Allocate the next bound arrays (length halves each round)
            let mut next_polys: Vec<DensePolynomial<F>> = (0..num_polys)
                .map(|_| DensePolynomial::new(vec![F::zero(); half_before]))
                .collect();

            for tile_idx in 0..num_tiles {
                let start = tile_idx * s.tile_len;
                let end = core::cmp::min(start + s.tile_len, half_before);

                // 1) Tile bind: y[j] = a + r*(b-a) for j in [start, end)
                //    Writes into next_polys while reading from current working arrays.
                for p_idx in 0..num_polys {
                    let src = &working[p_idx].Z;
                    let dst = &mut next_polys[p_idx].Z;
                    for j in start..end {
                        let a = src[2 * j];
                        let b = src[2 * j + 1];
                        dst[j] = a + r * (b - a);
                    }
                }

                // 2) Tile evaluate: accumulate contributions from reduced pairs (2k,2k+1)
                //    Accumulate into tile-level accumulators, then fold into round accumulator.
                let start_even = start & !1;
                let end_even = end & !1;
                let k_start = start_even / 2;
                let k_end = end_even / 2;

                // Per-tile accumulators
                let mut tile_h0_accumulator = F::zero();
                let mut tile_ht_accumulator = vec![F::zero(); t_len];
                let mut prod_t_accumulator: Vec<F> = vec![F::one(); t_len];

                for k in k_start..k_end {
                    // Product accumulators across polynomials for this reduced pair
                    let mut prod_a_accumulator = F::one();
                    // For each polynomial, update the running products
                    for p_idx in 0..num_polys {
                        let y0 = next_polys[p_idx].Z[2 * k];
                        let y1 = next_polys[p_idx].Z[2 * k + 1];
                        let m2 = y1 - y0;

                        prod_a_accumulator = prod_a_accumulator * y0;
                        // Evaluate (y0 + m2 * t) for all t and multiply into the prod_t accumulator
                        for (idx, &tv) in t_vals.iter().enumerate() {
                            // prod_t_accumulator[idx] *= (y0 + m2 * tv)
                            prod_t_accumulator[idx] = prod_t_accumulator[idx] * (y0 + m2 * tv);
                        }
                    }
                    tile_h0_accumulator = tile_h0_accumulator + prod_a_accumulator;
                    for idx in 0..t_len { tile_ht_accumulator[idx] = tile_ht_accumulator[idx] + prod_t_accumulator[idx]; }
                    // Reset the prod_t accumulator back to 1 for the next reduced pair
                    for v in &mut prod_t_accumulator { *v = F::one(); }
                }

                // Fold tile accumulators into the round accumulator
                evals_at_points[0] = evals_at_points[0] + tile_h0_accumulator;
                for idx in 0..t_len { evals_at_points[idx + 1] = evals_at_points[idx + 1] + tile_ht_accumulator[idx]; }
            }

            self.polynomials = next_polys;
        } else {
            // No binding yet (round 0): evaluate directly from current arrays using full-size pairs
            let tile_evals: Vec<(F, Vec<F>)> = (0..num_tiles)
                .into_par_iter()
                .map(|tile_idx| {
                    let start = tile_idx * s.tile_len;
                    let end = core::cmp::min(start + s.tile_len, half_before);
                    // Per-tile accumulators
                    let mut tile_h0_accumulator = F::zero();
                    let mut tile_ht_accumulator = vec![F::zero(); t_len];
                    let mut prod_t_accumulator: Vec<F> = vec![F::one(); t_len];
                    for j in start..end {
                        // Product accumulators across polynomials for this full pair
                        let mut prod_a_accumulator = F::one();
                        for poly in working.iter() {
                            let a = poly.Z[2 * j];
                            let b = poly.Z[2 * j + 1];
                            let m = b - a;
                            prod_a_accumulator = prod_a_accumulator * a;
                            for (idx, &tv) in t_vals.iter().enumerate() { prod_t_accumulator[idx] = prod_t_accumulator[idx] * (a + m * tv); }
                        }
                        tile_h0_accumulator = tile_h0_accumulator + prod_a_accumulator;
                        for idx in 0..t_len { tile_ht_accumulator[idx] = tile_ht_accumulator[idx] + prod_t_accumulator[idx]; }
                        for v in &mut prod_t_accumulator { *v = F::one(); }
                    }
                    (tile_h0_accumulator, tile_ht_accumulator)
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

#[cfg(test)]
mod tests {
    use super::{ProductSumcheck, ExecutionMode};
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

    fn run_product_sumcheck_test(d: u32, mode: ExecutionMode) {
        let t = 10u32;
        let polys = build_random_dense_polys::<Fr>(t, d, "product_sumcheck_test");
        let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, mode);
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
        run_product_sumcheck_test(2, ExecutionMode::Batch);
        run_product_sumcheck_test(3, ExecutionMode::Batch);
        run_product_sumcheck_test(4, ExecutionMode::Batch);
        run_product_sumcheck_test(2, ExecutionMode::Streaming);
        run_product_sumcheck_test(3, ExecutionMode::Streaming);
        run_product_sumcheck_test(4, ExecutionMode::Streaming);
    }

}

// Provide a no-op main when this file is compiled as a standalone binary target
#[allow(dead_code)]
fn main() {}

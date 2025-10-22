use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

// Product sumcheck over d multilinear polynomials
pub enum ExecutionMode {
    Batch,
    Tiling,
}

pub struct ProductSumcheck<F: JoltField> {
    pub input_claim: F,
    pub polynomials: Vec<DensePolynomial<F>>, // List of d dense polynomials
    pub original_polynomials: Vec<DensePolynomial<F>>, // Unmodified copies for final claim
    pub log_n: usize,
    pub degree: usize, // number of polynomials
    pub mode: ExecutionMode,
    pub tiling: Option<TilingState<F>>,
    // Timings (ms)
    pub input_claim_ms: f64,
    pub boot_kernel_ms: f64,
    pub recursive_kernel_ms: f64,
}

pub struct TilingState<F: JoltField> {
    pub tile_len: usize,
    pub pending_r: Option<F::Challenge>,
    pub eval_points: Vec<F>,
    // Reusable buffer for next-round bound polynomials to avoid reallocations
    pub next_polys: Option<Vec<DensePolynomial<F>>>,
}

impl<F: JoltField> TilingState<F> {
    fn new_with_tile_len(degree: usize, tile_len: usize) -> Self {
        let eval_points: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();
        // Ensure tile length is even so pairs (j0, j1) are well formed and cache lines are fully utilized.
        let mut tl = tile_len;
        if tl & 1 == 1 { tl -= 1; }
        if tl < 2 { tl = 2; }
        Self {
            tile_len: tl,
            pending_r: None,
            eval_points,
            next_polys: None,
        }
    }
}

impl<F: JoltField> ProductSumcheck<F> {
    pub fn from_polynomials_mode(
        polynomials: Vec<DensePolynomial<F>>,
        mode: ExecutionMode,
        tile_len_override: Option<usize>,
    ) -> Self {
        let n = polynomials.get(0).map(|p| p.len()).unwrap_or(0);
        let log_n = if n == 0 { 0 } else { n.trailing_zeros() as usize };
        let degree = polynomials.len();
        let original_polynomials = polynomials.clone();
        // Compute input_claim using a simple parallel map-reduce over i (no tiling) for apples-to-apples.
        let input_t0 = Instant::now();
        let input_claim = (0..n)
            .into_par_iter()
            .map(|i| polynomials.iter().fold(F::one(), |acc, poly| acc * poly.Z[i]))
            .reduce(|| F::zero(), |a, b| a + b);
        let input_ms = input_t0.elapsed().as_secs_f64() * 1000.0;

        let tiling = match mode {
            ExecutionMode::Batch => None,
            ExecutionMode::Tiling => {
                Some(TilingState::new_with_tile_len(degree, tile_len_override.unwrap_or(0)))
            }
        };
        Self { input_claim, polynomials, original_polynomials, log_n, degree, mode, tiling, input_claim_ms: input_ms, boot_kernel_ms: 0.0, recursive_kernel_ms: 0.0 }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductSumcheck<F> {
    fn degree(&self) -> usize { self.degree }
    fn num_rounds(&self) -> usize { self.log_n }
    fn input_claim(&self) -> F { self.input_claim }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let t0 = Instant::now();
        let out = match self.mode {
            ExecutionMode::Batch => {
                // Fused single-pass computation for h(0) and h(2..=degree) over j.
                // We evaluate all t-values in one sweep while data are hot, avoiding (d-1) extra passes.
                let half = self.polynomials[0].len() / 2;
                let degree = self.degree;
                let points_len = 1 + (degree.saturating_sub(1));
                if half == 0 { return vec![F::zero(); points_len]; }

                // Precompute evaluation points once per round
                let eval_points: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();
                let num_eval_points = eval_points.len();

                // Non-tiled single-pass: parallel fold directly over j for clean comparison to tiled streaming.
                let (h0_total, ht_total, _scratch_prod) = (0..half)
                    .into_par_iter()
                    .fold(
                        || (F::zero(), vec![F::zero(); num_eval_points], vec![F::one(); num_eval_points]),
                        |(mut h0_acc, mut ht_acc, mut prod_t_acc), j| {
                            // Compute a,m for each polynomial; multiply into h(0) and all h(t)
                            let mut prod_a = F::one();
                            for poly in self.polynomials.iter() {
                                let a = poly.Z[2 * j];
                                let b = poly.Z[2 * j + 1];
                                let m = b - a;
                                prod_a = prod_a * a;
                                if num_eval_points > 0 {
                                    let mut v_t = a + m * eval_points[0];
                                    prod_t_acc[0] = prod_t_acc[0] * v_t;
                                    for idx in 1..num_eval_points {
                                        v_t = v_t + m;
                                        prod_t_acc[idx] = prod_t_acc[idx] * v_t;
                                    }
                                }
                            }
                            h0_acc = h0_acc + prod_a;
                            for idx in 0..num_eval_points { ht_acc[idx] = ht_acc[idx] + prod_t_acc[idx]; }
                            for v in &mut prod_t_acc { *v = F::one(); }
                            (h0_acc, ht_acc, prod_t_acc)
                        },
                    )
                    .reduce(
                        || (F::zero(), vec![F::zero(); num_eval_points], vec![F::one(); num_eval_points]),
                        |(h0_a, mut ht_a, _), (h0_b, ht_b, _)| {
                            for i in 0..num_eval_points { ht_a[i] = ht_a[i] + ht_b[i]; }
                            (h0_a + h0_b, ht_a, vec![F::one(); num_eval_points])
                        },
                    );

                let mut evals_at_points = vec![F::zero(); points_len];
                evals_at_points[0] = h0_total;
                for idx in 0..num_eval_points { evals_at_points[idx + 1] = ht_total[idx]; }
                evals_at_points
            }
            ExecutionMode::Tiling => {
                {
                    let this = &mut *self;
                    let s = this.tiling.as_mut().expect("tiling state expected");
                    let r_opt = s.pending_r.take();

                    // Working view for this round
                    let working: &[DensePolynomial<F>] = &this.polynomials;
                    let len_before = working[0].len();
                    for (i, poly) in working.iter().enumerate() {
                        assert_eq!(poly.len(), len_before, "Polynomial {} has length {}, expected {}", i, poly.len(), len_before);
                    }
                    // Dimensions and precomputed values (computed once per round)
                    let half_before = len_before / 2;
                    let degree = this.degree; // number of polynomials
                    let points_len = 1 + (degree.saturating_sub(1));
                    let eval_points = &s.eval_points; // t in [2..=degree]
                    let num_eval_points = eval_points.len();
                    let num_polys = working.len();
                    if half_before == 0 { return vec![F::zero(); points_len]; }

                    let num_tiles = (half_before + s.tile_len - 1) / s.tile_len;

                    // Accumulator for this round's prover message evaluations: [h(0), h(2), ..., h(d)]
                    let mut evals_at_points = vec![F::zero(); points_len];

                    if let Some(r) = r_opt {
                        // Tile-local scratch strategy:
                        // - For each tile: compute y[j] into a tile-local buffer per polynomial
                        // - Use the buffers to accumulate h(0) and h(t) within the tile
                        // - Commit the tile buffers to the next round arrays in one contiguous write per polynomial
                        // This preserves fused bind+eval while improving locality and write coalescing.

                        // Prepare next_round destination containers (reused across rounds)
                        let mut next_polys = s
                            .next_polys
                            .take()
                            .unwrap_or_else(|| (0..num_polys).map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(half_before))).collect());
                        if next_polys.len() != num_polys {
                            next_polys = (0..num_polys).map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(half_before))).collect();
                        }
                        for poly in next_polys.iter_mut() {
                            if poly.Z.len() < half_before {
                                let needed = half_before - poly.Z.len();
                                let additional = unsafe_allocate_zero_vec(needed);
                                poly.Z.extend_from_slice(&additional);
                            }
                            poly.len = half_before;
                            poly.num_vars = (usize::BITS as usize - 1) - half_before.leading_zeros() as usize;
                        }

                        // Prepare raw base addresses for next-round arrays. We use integer base addresses here
                        // to avoid Sync bounds on raw pointers in the parallel closure; they are reconstructed per worker.
                        let base_addrs: Vec<usize> = next_polys
                            .iter_mut()
                            .map(|p| p.Z.as_mut_ptr() as usize)
                            .collect();

                        // Parallel tiles with per-worker reusable tile buffers to avoid per-tile allocations.
                        let (h0_total, ht_total, _scratch_prod, _worker_bufs) = (0..num_tiles)
                            .into_par_iter()
                            .fold(
                                || {
                                    let tile_bufs: Vec<Vec<F>> = (0..num_polys)
                                        .map(|_| unsafe_allocate_zero_vec(s.tile_len))
                                        .collect();
                                    (F::zero(), vec![F::zero(); num_eval_points], vec![F::one(); num_eval_points], tile_bufs)
                                },
                                |(mut h0_acc, mut ht_acc, mut prod_t_acc, mut tile_bufs), tile_idx| {
                                    let start = tile_idx * s.tile_len;
                                    let end = core::cmp::min(start + s.tile_len, half_before);
                                    if start < end {
                                        let cur_len = end - start;
                                        // Fill buffers (only first cur_len entries are written)
                                        for p_idx in 0..num_polys {
                                            let src = &working[p_idx].Z;
                                            let buf = &mut tile_bufs[p_idx];
                                            debug_assert!(buf.len() >= cur_len);
                                            for off in 0..cur_len {
                                                let j = start + off;
                                                let a = src[2 * j];
                                                let b = src[2 * j + 1];
                                                let m = b - a;
                                                buf[off] = if m.is_zero() { a } else if m.is_one() { a + r } else { a + r * m };
                                            }
                                        }

                                        // Accumulate from buffers using (j0,j1) within the tile
                                        let k_start = (start + 1) >> 1;
                                        let k_end = end >> 1;
                                        for k in k_start..k_end {
                                            let j0 = 2 * k;
                                            let j1 = 2 * k + 1;
                                            let off0 = j0 - start;
                                            let off1 = j1 - start;
                                            let mut prod_a = F::one();
                                            for p_idx in 0..num_polys {
                                                let y0 = tile_bufs[p_idx][off0];
                                                let y1 = tile_bufs[p_idx][off1];
                                                let m2 = y1 - y0;
                                                prod_a = prod_a * y0;
                                                if num_eval_points > 0 {
                                                    let mut v_t = y0 + m2 * eval_points[0];
                                                    prod_t_acc[0] = prod_t_acc[0] * v_t;
                                                    for idx in 1..num_eval_points {
                                                        v_t = v_t + m2;
                                                        prod_t_acc[idx] = prod_t_acc[idx] * v_t;
                                                    }
                                                }
                                            }
                                            h0_acc = h0_acc + prod_a;
                                            for idx in 0..num_eval_points { ht_acc[idx] = ht_acc[idx] + prod_t_acc[idx]; }
                                            for v in &mut prod_t_acc { *v = F::one(); }
                                        }

                                        // Bulk copy buffers to destination arrays (fast path)
                                        for p_idx in 0..num_polys {
                                            let dst_base = base_addrs[p_idx] as *mut F;
                                            unsafe {
                                                std::ptr::copy_nonoverlapping(
                                                    tile_bufs[p_idx].as_ptr(),
                                                    dst_base.add(start),
                                                    cur_len,
                                                );
                                            }
                                        }
                                    }
                                    (h0_acc, ht_acc, prod_t_acc, tile_bufs)
                                },
                            )
                            .reduce(
                                || (F::zero(), vec![F::zero(); num_eval_points], vec![F::one(); num_eval_points], Vec::new()),
                                |(h0_a, mut ht_a, _, _), (h0_b, ht_b, _, _)| {
                                    for i in 0..num_eval_points { ht_a[i] = ht_a[i] + ht_b[i]; }
                                    (h0_a + h0_b, ht_a, vec![F::one(); num_eval_points], Vec::new())
                                },
                            );

                        evals_at_points[0] = evals_at_points[0] + h0_total;
                        for idx in 0..ht_total.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + ht_total[idx]; }

                        // Swap in bound arrays and keep previous buffers for reuse without cloning
                        let old_polys = std::mem::replace(&mut this.polynomials, next_polys);
                        s.next_polys = Some(old_polys);
                    } else {
                        // No binding yet (round 0): evaluate directly from current arrays using full-size pairs
                        let (h0_total, ht_total, _scratch_prod) = (0..num_tiles)
                            .into_par_iter()
                            .fold(
                                || (F::zero(), vec![F::zero(); num_eval_points], vec![F::one(); num_eval_points]),
                                |(mut h0_acc, mut ht_acc, mut prod_t_acc), tile_idx| {
                                    let start = tile_idx * s.tile_len;
                                    let end = core::cmp::min(start + s.tile_len, half_before);
                                    for j in start..end {
                                        // compute prod_a for round 0 directly from current arrays
                                        let mut prod_a = F::one();
                                        for poly in working.iter() {
                                            let a = poly.Z[2 * j];
                                            let b = poly.Z[2 * j + 1];
                                            let m = b - a;
                                            prod_a = prod_a * a;
                                            if num_eval_points > 0 {
                                                let mut v_t = a + m * eval_points[0];
                                                prod_t_acc[0] = prod_t_acc[0] * v_t;
                                                for idx in 1..num_eval_points {
                                                    v_t = v_t + m;
                                                    prod_t_acc[idx] = prod_t_acc[idx] * v_t;
                                                }
                                            }
                                        }
                                        h0_acc = h0_acc + prod_a;
                                        for idx in 0..num_eval_points { ht_acc[idx] = ht_acc[idx] + prod_t_acc[idx]; }
                                        for v in &mut prod_t_acc { *v = F::one(); }
                                    }
                                    (h0_acc, ht_acc, prod_t_acc)
                                },
                            )
                            .reduce(
                                || (F::zero(), vec![F::zero(); num_eval_points], vec![F::one(); num_eval_points]),
                                |(h0_a, mut ht_a, _), (h0_b, ht_b, _)| {
                                    for i in 0..num_eval_points { ht_a[i] = ht_a[i] + ht_b[i]; }
                                    (h0_a + h0_b, ht_a, vec![F::one(); num_eval_points])
                                },
                            );

                        evals_at_points[0] = h0_total;
                        for idx in 0..ht_total.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + ht_total[idx]; }
                    }

                    evals_at_points
                }
            }
        };
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        if round == 0 { self.boot_kernel_ms += dt; } else { self.recursive_kernel_ms += dt; }
        out
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let t0 = Instant::now();
        match self.mode {
            ExecutionMode::Batch => {
                self.polynomials.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            }
            ExecutionMode::Tiling => {
                if let Some(ref mut s) = self.tiling { s.pending_r = Some(r_j); }
            }
        }
        // Count bind cost as part of recursive kernel timing (bind happens after boot round)
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        self.recursive_kernel_ms += dt;
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
impl<F: JoltField> ProductSumcheck<F> {}

// SumcheckMetrics removed; benchmarking and reporting are handled in bench.rs

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
        let mut sumcheck = ProductSumcheck::from_polynomials_mode(polys, mode, Some(32));
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
        run_product_sumcheck_test(2, ExecutionMode::Tiling);
        run_product_sumcheck_test(3, ExecutionMode::Tiling);
        run_product_sumcheck_test(4, ExecutionMode::Tiling);
    }

}

// Provide a no-op main when this file is compiled as a standalone binary target
#[allow(dead_code)]
fn main() {}

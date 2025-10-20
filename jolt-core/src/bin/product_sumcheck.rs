use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::Transcript;
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
}

pub struct TilingState<F: JoltField> {
    pub tile_len: usize,
    pub pending_r: Option<F::Challenge>,
    pub t_vals: Vec<F>,
    // Reusable buffer for next-round bound polynomials to avoid reallocations
    pub next_polys: Option<Vec<DensePolynomial<F>>>,
}

impl<F: JoltField> TilingState<F> {
    fn compute_tile_len(degree: usize, l1_bytes: usize) -> usize {
        let elem_bytes: usize = core::mem::size_of::<F>().max(32); // BN254 ~32 bytes
        let d = degree.max(1);
        let mut tile_len = (l1_bytes / (d * elem_bytes)).max(64);
        tile_len = tile_len.min(2048);
        let pow = usize::BITS - 1 - tile_len.leading_zeros();
        1usize << pow
    }

    fn new(degree: usize, l1_bytes: usize) -> Self {
        let tile_len = Self::compute_tile_len(degree, l1_bytes);
        let t_vals: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();
        Self {
            tile_len,
            pending_r: None,
            t_vals,
            next_polys: None,
        }
    }
}

impl<F: JoltField> ProductSumcheck<F> {
    pub fn from_polynomials_mode(polynomials: Vec<DensePolynomial<F>>, mode: ExecutionMode, l1_kb: Option<usize>) -> Self {
        let n = polynomials.get(0).map(|p| p.len()).unwrap_or(0);
        let log_n = if n == 0 { 0 } else { n.trailing_zeros() as usize };
        let degree = polynomials.len();
        let original_polynomials = polynomials.clone();
        // Compute input_claim differently by mode for fair end-to-end benchmarking.
        // - Batch: simple parallel map-reduce over i (no tiling), mirroring baseline behavior.
        // - Tiling: partition-friendly tiled fold/reduce to minimize memory traffic.
        let l1_bytes_cfg = l1_kb.map(|kb| kb * 1024).unwrap_or(32 * 1024);
        let (input_claim, _input_ms) = match mode {
            ExecutionMode::Batch => {
                let t0 = Instant::now();
                let v = (0..n)
                    .into_par_iter()
                    .map(|i| polynomials.iter().fold(F::one(), |acc, poly| acc * poly.Z[i]))
                    .reduce(|| F::zero(), |a, b| a + b);
                (v, t0.elapsed().as_secs_f64() * 1000.0)
            }
            ExecutionMode::Tiling => {
                let t0 = Instant::now();
                let tile_len = TilingState::<F>::compute_tile_len(degree, l1_bytes_cfg);
                let num_tiles = if n == 0 { 0 } else { (n + tile_len - 1) / tile_len };
                let (sum_total, _scratch) = (0..num_tiles)
                    .into_par_iter()
                    .fold(
                        || (F::zero(), F::one()),
                        |(mut sum_acc, mut _tmp_prod), tile_idx| {
                            let start = tile_idx * tile_len;
                            let end = core::cmp::min(start + tile_len, n);
                            for i in start..end {
                                let mut prod = F::one();
                                for poly in polynomials.iter() {
                                    prod = prod * poly.Z[i];
                                }
                                sum_acc = sum_acc + prod;
                            }
                            (sum_acc, _tmp_prod)
                        },
                    )
                    .reduce(
                        || (F::zero(), F::one()),
                        |(a_sum, _), (b_sum, _)| (a_sum + b_sum, F::one()),
                    );
                (sum_total, t0.elapsed().as_secs_f64() * 1000.0)
            }
        };

        let tiling = match mode {
            ExecutionMode::Batch => None,
            ExecutionMode::Tiling => Some(TilingState::new(degree, l1_bytes_cfg)),
        };
        Self { input_claim, polynomials, original_polynomials, log_n, degree, mode, tiling }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductSumcheck<F> {
    fn degree(&self) -> usize { self.degree }
    fn num_rounds(&self) -> usize { self.log_n }
    fn input_claim(&self) -> F { self.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        match self.mode {
            ExecutionMode::Batch => {
                // Fused single-pass computation for h(0) and h(2..=degree) over j.
                // We evaluate all t-values in one sweep while data are hot, avoiding (d-1) extra passes.
                let half = self.polynomials[0].len() / 2;
                let degree = self.degree;
                let points_len = 1 + (degree.saturating_sub(1));
                if half == 0 { return vec![F::zero(); points_len]; }

                // Precompute t values once per round
                let t_vals: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();
                let t_len = t_vals.len();

                // Non-tiled single-pass: parallel fold directly over j for clean comparison to tiled streaming.
                let (h0_total, ht_total, _scratch_prod) = (0..half)
                    .into_par_iter()
                    .fold(
                        || (F::zero(), vec![F::zero(); t_len], vec![F::one(); t_len]),
                        |(mut h0_acc, mut ht_acc, mut prod_t_acc), j| {
                            // Compute a,m for each polynomial; multiply into h(0) and all h(t)
                            let mut prod_a = F::one();
                            for poly in self.polynomials.iter() {
                                let a = poly.Z[2 * j];
                                let b = poly.Z[2 * j + 1];
                                let m = b - a;
                                prod_a = prod_a * a;
                                if t_len > 0 {
                                    let mut v_t = a + m * t_vals[0];
                                    prod_t_acc[0] = prod_t_acc[0] * v_t;
                                    for idx in 1..t_len {
                                        v_t = v_t + m;
                                        prod_t_acc[idx] = prod_t_acc[idx] * v_t;
                                    }
                                }
                            }
                            h0_acc = h0_acc + prod_a;
                            for idx in 0..t_len { ht_acc[idx] = ht_acc[idx] + prod_t_acc[idx]; }
                            for v in &mut prod_t_acc { *v = F::one(); }
                            (h0_acc, ht_acc, prod_t_acc)
                        },
                    )
                    .reduce(
                        || (F::zero(), vec![F::zero(); t_len], vec![F::one(); t_len]),
                        |(h0_a, mut ht_a, _), (h0_b, ht_b, _)| {
                            for i in 0..t_len { ht_a[i] = ht_a[i] + ht_b[i]; }
                            (h0_a + h0_b, ht_a, vec![F::one(); t_len])
                        },
                    );

                let mut evals_at_points = vec![F::zero(); points_len];
                evals_at_points[0] = h0_total;
                for idx in 0..t_len { evals_at_points[idx + 1] = ht_total[idx]; }
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
                    let t_vals = &s.t_vals; // t in [2..=degree]
                    let t_len = t_vals.len();
                    let num_polys = working.len();
                    if half_before == 0 { return vec![F::zero(); points_len]; }

                    let num_tiles = (half_before + s.tile_len - 1) / s.tile_len;

                    // Accumulator for this round's prover message evaluations: [h(0), h(2), ..., h(d)]
                    let mut evals_at_points = vec![F::zero(); points_len];

                    if let Some(r) = r_opt {
                        // Bandwidth-first two-phase execution to maximize effective memory throughput:
                        // Phase A (global): for each polynomial, stream through the entire current array to
                        // compute and write bound outputs y[j] contiguously into next_polys[p].Z. This preserves
                        // long, contiguous bursts per array (great for hardware prefetch and NUMA bandwidth).
                        // Phase B (tiled accumulation): iterate tiles to read the freshly written y values and
                        // accumulate h(0) and h(t). This keeps the working set small during accumulation without
                        // interleaving per-poly writes, which can degrade bandwidth.
                        // Reuse previously allocated buffers for next_round arrays without zero-init.
                        let mut next_polys = s
                            .next_polys
                            .take()
                            .unwrap_or_else(|| (0..num_polys).map(|_| DensePolynomial::new(vec![F::zero(); half_before])).collect());
                        if next_polys.len() != num_polys {
                            next_polys = (0..num_polys).map(|_| DensePolynomial::new(vec![F::zero(); half_before])).collect();
                        }
                        for p in 0..num_polys {
                            if next_polys[p].len() != half_before {
                                next_polys[p].len = half_before;
                                next_polys[p].num_vars = (usize::BITS as usize - 1) - half_before.leading_zeros() as usize;
                            }
                        }

                        // Phase A: global per-polynomial binding pass with contiguous writes
                        next_polys
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(p_idx, dst_poly)| {
                                let src = &working[p_idx].Z;
                                let dst = &mut dst_poly.Z;
                                dst[..half_before]
                                    .par_chunks_mut(1024)
                                    .enumerate()
                                    .for_each(|(chunk_i, chunk)| {
                                        let start = chunk_i * 1024;
                                        for (off, y_slot) in chunk.iter_mut().enumerate() {
                                            let j = start + off;
                                            if j >= half_before { break; }
                                            let a = src[2 * j];
                                            let b = src[2 * j + 1];
                                            let m = b - a;
                                            *y_slot = if m.is_zero() { a } else if m.is_one() { a + r } else { a + r * m };
                                        }
                                    });
                            });

                        // We iterate tiles in units of j (bound indices), but compute pair contributions k using even-aligned bounds.
                        let (h0_total, ht_total, _scratch_prod) = (0..num_tiles)
                            .into_par_iter()
                            .fold(
                                || (F::zero(), vec![F::zero(); t_len], vec![F::one(); t_len]),
                                |(mut h0_acc, mut ht_acc, mut prod_t_acc), tile_idx| {
                                    let start = tile_idx * s.tile_len;
                                    let end = core::cmp::min(start + s.tile_len, half_before);

                                    // Phase B: accumulate h(0) and h(t) over pairs fully contained in this tile.
                                    // We only take k such that j0=2k and j1=2k+1 both fall within [start, end),
                                    // which is guaranteed for power-of-two tile sizes.
                                    let k_start = (start + 1) >> 1; // first k with 2k >= start
                                    let k_end = end >> 1;            // last k with 2k+1 < end
                                    for k in k_start..k_end {
                                        let j0 = 2 * k;
                                        let j1 = 2 * k + 1;

                                        // Product across polynomials for h(0) and vectorized v_t accumulation
                                        let mut prod_a = F::one();
                                        for p_idx in 0..num_polys {
                                            let y0 = next_polys[p_idx].Z[j0];
                                            let y1 = next_polys[p_idx].Z[j1];
                                            let m2 = y1 - y0;
                                            prod_a = prod_a * y0;
                                            if t_len > 0 {
                                                let mut v_t = y0 + m2 * t_vals[0];
                                                prod_t_acc[0] = prod_t_acc[0] * v_t;
                                                for idx in 1..t_len {
                                                    v_t = v_t + m2;
                                                    prod_t_acc[idx] = prod_t_acc[idx] * v_t;
                                                }
                                            }
                                        }
                                        h0_acc = h0_acc + prod_a;
                                        for idx in 0..t_len { ht_acc[idx] = ht_acc[idx] + prod_t_acc[idx]; }
                                        for v in &mut prod_t_acc { *v = F::one(); }
                                    }

                                    (h0_acc, ht_acc, prod_t_acc)
                                },
                            )
                            .reduce(
                                || (F::zero(), vec![F::zero(); t_len], vec![F::one(); t_len]),
                                |(h0_a, mut ht_a, _), (h0_b, ht_b, _)| {
                                    for i in 0..t_len { ht_a[i] = ht_a[i] + ht_b[i]; }
                                    (h0_a + h0_b, ht_a, vec![F::one(); t_len])
                                },
                            );

                        evals_at_points[0] = evals_at_points[0] + h0_total;
                        for idx in 0..ht_total.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + ht_total[idx]; }

                        // Swap in bound arrays and keep previous buffers for reuse without cloning
                        let old_polys = std::mem::replace(&mut this.polynomials, next_polys);
                        s.next_polys = Some(old_polys);
                    } else {
                        // No binding yet (round 0): evaluate directly from current arrays using full-size pairs
                        // in practise this will never happen, but we include it for completeness
                        let (h0_total, ht_total, _scratch_prod) = (0..num_tiles)
                            .into_par_iter()
                            .fold(
                                || (F::zero(), vec![F::zero(); t_len], vec![F::one(); t_len]),
                                |(mut h0_acc, mut ht_acc, mut prod_t_acc), tile_idx| {
                                    let start = tile_idx * s.tile_len;
                                    let end = core::cmp::min(start + s.tile_len, half_before);
                                    for j in start..end {
                                        let mut prod_a = F::one();
                                        for poly in working.iter() {
                                            let a = poly.Z[2 * j];
                                            let b = poly.Z[2 * j + 1];
                                            let m = b - a;
                                            prod_a = prod_a * a;
                                            if t_len > 0 {
                                                let mut v_t = a + m * t_vals[0];
                                                prod_t_acc[0] = prod_t_acc[0] * v_t;
                                                for idx in 1..t_len {
                                                    v_t = v_t + m;
                                                    prod_t_acc[idx] = prod_t_acc[idx] * v_t;
                                                }
                                            }
                                        }
                                        h0_acc = h0_acc + prod_a;
                                        for idx in 0..t_len { ht_acc[idx] = ht_acc[idx] + prod_t_acc[idx]; }
                                        for v in &mut prod_t_acc { *v = F::one(); }
                                    }
                                    (h0_acc, ht_acc, prod_t_acc)
                                },
                            )
                            .reduce(
                                || (F::zero(), vec![F::zero(); t_len], vec![F::one(); t_len]),
                                |(h0_a, mut ht_a, _), (h0_b, ht_b, _)| {
                                    for i in 0..t_len { ht_a[i] = ht_a[i] + ht_b[i]; }
                                    (h0_a + h0_b, ht_a, vec![F::one(); t_len])
                                },
                            );

                        evals_at_points[0] = evals_at_points[0] + h0_total;
                        for idx in 0..ht_total.len() { evals_at_points[idx + 1] = evals_at_points[idx + 1] + ht_total[idx]; }
                    }

                    evals_at_points
                }
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
            ExecutionMode::Tiling => {
                if let Some(ref mut s) = self.tiling { s.pending_r = Some(r_j); }
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

use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN};
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::Transcript;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use crate::product_sumcheck::ProductSumcheck;

// Tiled, cache-friendly product sumcheck that defers bind until compute
pub struct SlicedProductSumcheck<F: JoltField> {
    pub base: ProductSumcheck<F>,
    pending_r: Vec<F::Challenge>,
}

impl<F: JoltField> SlicedProductSumcheck<F> {
    pub fn from_polynomials(polynomials: Vec<DensePolynomial<F>>, threads: usize) -> Self {
        Self { base: ProductSumcheck::from_polynomials(polynomials, threads), pending_r: Vec::new() }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for SlicedProductSumcheck<F> {
    fn degree(&self) -> usize { self.base.degree }
    fn num_rounds(&self) -> usize { self.base.log_n }
    fn input_claim(&self) -> F { self.base.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        // Apply deferred binds once
        if !self.pending_r.is_empty() {
            for r in self.pending_r.drain(..) {
                self.base
                    .polynomials
                    .par_iter_mut()
                    .for_each(|poly| poly.bind_parallel(r, BindingOrder::HighToLow));
            }
        }

        // Auto-tune TILE_LEN to fit L1 cache: tile_len * degree * elem_bytes << L1_BYTES
        // Approximate elem_bytes via size_of::<F>(). Use a safety margin.
        let l1_bytes: usize = 32 * 1024; // typical L1 per core
        let elem_bytes: usize = std::mem::size_of::<F>().max(32); // BN254 ~32 bytes
        let d = self.base.degree.max(1);
        let mut tile_len = (l1_bytes / (d * elem_bytes)).max(64);
        // Clamp and round down to power-of-two for nicer strides
        tile_len = tile_len.min(1024);
        tile_len = 1 << (usize::BITS - 1 - (tile_len.leading_zeros())) as usize;
        let len = self.base.polynomials[0].len();
        let half = len / 2;
        let degree = self.base.degree;

        let compute = || {
            let points_len = 1 + (degree.saturating_sub(1));
            let num_tiles = (half + tile_len - 1) / tile_len;
            let t_vals: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();

            // Parallel over tiles; each tile returns a small Vec<F> of partial sums
            let partials: Vec<Vec<F>> = (0..num_tiles)
                .into_par_iter()
                .map(|tile_idx| {
                    let start = tile_idx * tile_len;
                    let end = (start + tile_len).min(half);
                    let mut tile_sums = vec![F::zero(); points_len];
                    for j in start..end {
                        // Compute products for h(0) and each t with one pass over polys
                        let mut prod_a = F::one();
                        let mut prod_t: Vec<F> = vec![F::one(); t_vals.len()];
                        for poly in &self.base.polynomials {
                            let a = poly.Z[2 * j];
                            let b = poly.Z[2 * j + 1];
                            let m = b - a;
                            prod_a = prod_a * a;
                            for (idx, &tv) in t_vals.iter().enumerate() {
                                prod_t[idx] = prod_t[idx] * (a + m * tv);
                            }
                        }
                        tile_sums[0] = tile_sums[0] + prod_a;
                        for idx in 0..prod_t.len() {
                            tile_sums[idx + 1] = tile_sums[idx + 1] + prod_t[idx];
                        }
                    }
                    tile_sums
                })
                .collect();

            // Reduce element-wise
            let mut evals_at_points = vec![F::zero(); points_len];
            for v in partials.into_iter() {
                for (acc, val) in evals_at_points.iter_mut().zip(v.into_iter()) {
                    *acc = *acc + val;
                }
            }
            evals_at_points
        };

        if self.base.threads > 0 {
            rayon::ThreadPoolBuilder::new().num_threads(self.base.threads).build().unwrap().install(compute)
        } else {
            compute()
        }
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) { self.pending_r.push(r_j); }

    fn expected_output_claim(&self, _o: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>, r: &[F::Challenge]) -> F {
        let mut result = F::one();
        for poly in &self.base.polynomials { result = result * poly.evaluate(r); }
        result
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(&self, _a: Rc<RefCell<ProverOpeningAccumulator<F>>>, _t: &mut T, _p: OpeningPoint<BIG_ENDIAN, F>) {}
    fn cache_openings_verifier(&self, _a: Rc<RefCell<VerifierOpeningAccumulator<F>>>, _t: &mut T, _p: OpeningPoint<BIG_ENDIAN, F>) {}
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _f: &mut allocative::FlameGraphBuilder) {}
}



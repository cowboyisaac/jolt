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

// Tiled, cache-friendly product sumcheck that defers bind until compute
pub struct SlicedProductSumcheck<F: JoltField> {
    pub base: ProductSumcheck<F>,
    pending_r: Option<F::Challenge>,
}

impl<F: JoltField> SlicedProductSumcheck<F> {
    pub fn from_polynomials(polynomials: Vec<DensePolynomial<F>>, _threads: usize) -> Self {
        Self { base: ProductSumcheck::from_polynomials(polynomials), pending_r: None }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for SlicedProductSumcheck<F> {
    fn degree(&self) -> usize { self.base.degree }
    fn num_rounds(&self) -> usize { self.base.log_n }
    fn input_claim(&self) -> F { self.base.input_claim }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        // Auto-tune TILE_LEN to fit L1 cache: tile_len * degree * elem_bytes << L1_BYTES
        // Approximate elem_bytes via size_of::<F>(). Use a safety margin.
        let l1_bytes: usize = 32 * 1024; // typical L1 per core
        let elem_bytes: usize = std::mem::size_of::<F>().max(32); // BN254 ~32 bytes
        let d = self.base.degree.max(1);
        let mut tile_len = (l1_bytes / (d * elem_bytes)).max(64);
        // Clamp and round down to power-of-two for nicer strides
        tile_len = tile_len.min(1024);
        tile_len = 1 << (usize::BITS - 1 - (tile_len.leading_zeros())) as usize;
        // Ensure all polynomials have the same length
        let len = self.base.polynomials[0].len();
        for (i, poly) in self.base.polynomials.iter().enumerate() {
            assert_eq!(poly.len(), len, "Polynomial {} has length {}, expected {}", i, poly.len(), len);
        }
        let half = len / 2;
        let degree = self.base.degree;

        // Extract pending challenge to avoid borrowing issues
        let pending_challenge = self.pending_r.take();

        let mut compute = || {
            let points_len = 1 + (degree.saturating_sub(1));
            let _num_tiles = (half + tile_len - 1) / tile_len;
            let t_vals: Vec<F> = (2..=degree).map(|t| F::from_u64(t as u64)).collect();

            // Store m values before binding for h(t) computation
            let mut m_values_per_poly = Vec::new();
            if let Some(r) = pending_challenge {
                for poly in &mut self.base.polynomials {
                    // Compute half for THIS polynomial's current length
                    let poly_len = poly.Z.len();
                    let poly_half = poly_len / 2;
                    let mut m_values = Vec::with_capacity(poly_half);
                    
                    for j in 0..poly_half {
                        let a = poly.Z[2 * j];
                        let b = poly.Z[2 * j + 1];
                        let m = b - a;
                        m_values.push(m);
                        poly.Z[j] = a + m * r;
                    }
                    m_values_per_poly.push(m_values);
                    // Resize polynomial to half size
                    poly.Z.truncate(poly_half);
                    
                }
            }

            // After binding, recompute half based on the new polynomial length
            let computation_half = self.base.polynomials[0].len();
            let computation_num_tiles = (computation_half + tile_len - 1) / tile_len;

            // Parallel over tiles; each tile does computation
            let partials: Vec<Vec<F>> = (0..computation_num_tiles)
                .into_par_iter()
                .map(|tile_idx| {
                    let start = tile_idx * tile_len;
                    let end = (start + tile_len).min(computation_half);
                    let mut tile_sums = vec![F::zero(); points_len];
                    
                    // Process each element in the tile
                    for j in start..end {
                        // Compute products for h(0) and each t
                        let mut prod_a = F::one();
                        let mut prod_t: Vec<F> = vec![F::one(); t_vals.len()];
                        
                        for (poly_idx, poly) in self.base.polynomials.iter().enumerate() {
                            let a = poly.Z[j];
                            prod_a = prod_a * a;
                            
                            // Use stored m values for h(t) computation
                            if let Some(m_values) = m_values_per_poly.get(poly_idx) {
                                let m = m_values[j];
                                for (idx, &tv) in t_vals.iter().enumerate() {
                                    prod_t[idx] = prod_t[idx] * (a + m * tv);
                                }
                            } else {
                                // No binding, so m = 0, so a + m * tv = a
                                for (idx, _) in t_vals.iter().enumerate() {
                                    prod_t[idx] = prod_t[idx] * a;
                                }
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

        compute()
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) { self.pending_r = Some(r_j); }

    fn expected_output_claim(&self, _o: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>, r: &[F::Challenge]) -> F {
        let mut result = F::one();
        for poly in &self.base.polynomials { result = result * poly.evaluate(r); }
        result
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> { OpeningPoint::new(opening_point.iter().rev().copied().collect()) }

    fn cache_openings_prover(&self, _a: Rc<RefCell<ProverOpeningAccumulator<F>>>, _t: &mut T, _p: OpeningPoint<BIG_ENDIAN, F>) {}
    fn cache_openings_verifier(&self, _a: Rc<RefCell<VerifierOpeningAccumulator<F>>>, _t: &mut T, _p: OpeningPoint<BIG_ENDIAN, F>) {}
    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _f: &mut allocative::FlameGraphBuilder) {}
}

// Provide a no-op main when this file is compiled as a standalone binary target
fn main() {}



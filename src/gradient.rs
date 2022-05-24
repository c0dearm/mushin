use arrayfire::{constant, Array};

use crate::context::function::Function;
use crate::tensor::{Origin, Tensor};

/// Stores the gradients for a given tensor
pub struct Gradients(Vec<Array<f32>>);

impl Gradients {
    /// Given a root tensor, computes all the derivatives with respect to each of the variables it depends on
    /// by performing reverse auto-differentiation on its computation graph
    #[must_use]
    #[inline]
    pub fn compute<const B: u64, const N: u64, const R: u64, const C: u64>(
        z: &Tensor<B, N, R, C>,
    ) -> Self {
        match *z.origin() {
            Origin::Function(x_fid) => {
                let mut gradients = vec![constant!(0.0; 1, 1, 1, 1); z.context().tape_len()];
                gradients[x_fid] = constant!(1.0; R, C, N, B);

                for (i, function) in z.context().functions().iter().enumerate().rev() {
                    match *function {
                        Function::Nary => {}
                        Function::Unary(ref f) => {
                            let (fid, partial) = f.backward(&gradients[i]);
                            gradients[fid] = &gradients[fid] + partial;
                        }
                        Function::Binary(ref f) => {
                            let (fid_a, partial_a, fid_b, partial_b) = f.backward(&gradients[i]);
                            if let Some(fa) = fid_a {
                                gradients[fa] = &gradients[fa] + partial_a;
                            }
                            if let Some(fb) = fid_b {
                                gradients[fb] = &gradients[fb] + partial_b;
                            }
                        }
                    }
                }
                Self(gradients)
            }
            Origin::None => Self(vec![]),
        }
    }

    /// Returns the gradient of the root tensor with respect to another tensor in the computation graph
    #[must_use]
    #[inline]
    pub fn wrt<'ctx, const B: u64, const L: u64, const R: u64, const C: u64>(
        &self,
        x: &'ctx Tensor<'ctx, B, L, R, C>,
    ) -> Tensor<'ctx, B, L, R, C> {
        let value: Array<f32> = match usize::try_from(x) {
            Ok(function) => self
                .0
                .get(function)
                .map_or_else(|| constant!(0.0; R, C, L, B), std::clone::Clone::clone),
            _ => constant!(0.0; R, C, L, B),
        };

        Tensor::new(value, Origin::None, x.context())
    }
}

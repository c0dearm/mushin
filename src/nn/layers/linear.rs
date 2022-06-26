use crate as mu;
use crate::graph::node::Node;
use crate::tensor::{constant::Constant, params::DoubleParam, variable::Variable, Tensor};
use arrayfire::{seq, view, Array, MatProp};
use std::rc::Rc;

/// A Linear (perceptron) neural network layer with `I` input size and `O` output size
pub struct Linear<const I: u64, const O: u64, W>(W);

impl<const I: u64, const O: u64, W> Linear<I, O, W> {
    #[inline]
    const fn new(weights: W) -> Self {
        Self(weights)
    }

    /// Given an input computes the output
    #[inline]
    pub fn forward<X>(&self, x: &X) -> X::Out
    where
        W: Tensor<BATCH = 1, CHANNELS = 1, HEIGHT = { I + 1 }, WIDTH = { O }>,
        X: Tensor<CHANNELS = 1, HEIGHT = 1, WIDTH = { I }> + DoubleParam<{ X::BATCH }, 1, 1, O, W>,
    {
        let padded = arrayfire::join(1, &x.data(), &arrayfire::constant!(1.0; 1, 1, 1, X::BATCH));

        let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
            let a = arrayfire::matmul(
                df,
                &args[1],
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::TRANS,
            );

            let b = arrayfire::matmul(
                &args[0],
                df,
                arrayfire::MatProp::TRANS,
                arrayfire::MatProp::NONE,
            );

            let all = seq!();
            let unpad = seq!(0:-2:1);
            (view!(a[all, unpad, all, all]), b)
        };
        x.push_binary(
            &self.0,
            arrayfire::matmul(&padded, &self.0.data(), MatProp::NONE, MatProp::NONE),
            reverse,
            &[padded, self.0.data()],
        )
    }
}

impl<const I: u64, const O: u64> Linear<I, O, Variable<1, 1, { I + 1 }, O>> {
    #[must_use]
    #[inline]
    pub fn randn() -> Self {
        Self::new(mu::randn())
    }

    /// Consumes this layer and returns a copy with constant parameters
    #[must_use]
    #[inline]
    pub fn freeze(self) -> Linear<I, O, Constant<1, 1, { I + 1 }, O>> {
        Linear(self.0.freeze())
    }

    /// Returns the layer's trainable parameters
    #[must_use]
    #[inline]
    pub fn parameters(&self) -> Rc<Node> {
        (&self.0).into()
    }
}

impl<const I: u64, const O: u64> Linear<I, O, Constant<1, 1, { I + 1 }, O>> {
    /// Consumes this layer and returns a copy with trainable parameters
    #[must_use]
    #[inline]
    pub fn unfreeze(self) -> Linear<I, O, Variable<1, 1, { I + 1 }, O>> {
        Linear(self.0.unfreeze())
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;
    use arrayfire::Array;

    #[test]
    fn linear_forward_backward() {
        let linear = Linear::<3, 5, _>::new(mu::fill::<1, 1, 4, 5>(1.0));
        let x = mu::fill::<1, 1, 1, 3>(0.5);

        let z = linear.forward(&x);
        assert!(equal_arrays(
            z.data(),
            arrayfire::constant!(2.5; 1, 5, 1, 1)
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(5.0; 1, 3, 1, 1)
        ));
        assert!(equal_arrays(
            linear.parameters().grad().clone(),
            Array::new(
                &[
                    0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0,
                    0.5, 0.5, 0.5, 1.0
                ],
                arrayfire::dim4!(4, 5, 1, 1)
            )
        ));
    }

    #[test]
    fn linear_freeze_unfreeze() {
        let linear = Linear::<3, 5, _>::randn();
        let linear = linear.freeze();
        let _ = linear.unfreeze();
    }
}

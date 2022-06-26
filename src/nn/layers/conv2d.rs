use crate as mu;
use crate::graph::node::Node;
use crate::tensor::{constant::Constant, params::DoubleParam, variable::Variable, Tensor};
use arrayfire::{dim4, Array, ConvGradientType};
use std::rc::Rc;

/// A 2 dimensional convolutional layer with `I` input channels, `O` output channels and `H` height and `W` width kernel size
pub struct Conv2D<const I: u64, const O: u64, const H: u64, const W: u64, K>(K);

impl<const I: u64, const O: u64, const H: u64, const W: u64, K> Conv2D<I, O, H, W, K> {
    #[inline]
    const fn new(kernel: K) -> Self {
        Self(kernel)
    }

    /// Given an input computes the output
    #[inline]
    pub fn forward<X>(&self, x: &X) -> X::Out
    where
        K: Tensor<BATCH = { O }, CHANNELS = { I }, HEIGHT = { H }, WIDTH = { W }>,
        X: Tensor<CHANNELS = { I }>
            + DoubleParam<
                { X::BATCH },
                { K::BATCH },
                { X::HEIGHT - K::HEIGHT + 1 },
                { X::WIDTH - K::WIDTH + 1 },
                K,
            >,
    {
        let result = arrayfire::convolve2_nn(
            &x.data(),
            &self.0.data(),
            dim4!(1, 1),
            dim4!(0, 0),
            dim4!(1, 1),
        );

        let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
            let (a, k, out) = (&args[0], &args[1], &args[2]);
            (
                arrayfire::convolve2_gradient_nn(
                    df,
                    a,
                    k,
                    out,
                    dim4!(1, 1),
                    dim4!(0, 0),
                    dim4!(1, 1),
                    ConvGradientType::DATA,
                ),
                arrayfire::convolve2_gradient_nn(
                    df,
                    a,
                    k,
                    out,
                    dim4!(1, 1),
                    dim4!(0, 0),
                    dim4!(1, 1),
                    ConvGradientType::FILTER,
                ),
            )
        };

        x.push_binary(
            &self.0,
            result.clone(),
            reverse,
            &[x.data(), self.0.data(), result],
        )
    }
}

impl<const I: u64, const O: u64, const H: u64, const W: u64>
    Conv2D<I, O, H, W, Variable<O, I, H, W>>
{
    #[must_use]
    #[inline]
    pub fn randn() -> Self {
        Self::new(mu::randn())
    }

    /// Consumes this layer and returns a copy with constant parameters
    #[must_use]
    #[inline]
    pub fn freeze(self) -> Conv2D<I, O, H, W, Constant<O, I, H, W>> {
        Conv2D(self.0.freeze())
    }

    /// Returns the layer's trainable parameters
    #[must_use]
    #[inline]
    pub fn parameters(&self) -> Rc<Node> {
        (&self.0).into()
    }
}

impl<const I: u64, const O: u64, const H: u64, const W: u64>
    Conv2D<I, O, H, W, Constant<O, I, H, W>>
{
    /// Consumes this layer and returns a copy with trainable parameters
    #[must_use]
    #[inline]
    pub fn unfreeze(self) -> Conv2D<I, O, H, W, Variable<O, I, H, W>> {
        Conv2D(self.0.unfreeze())
    }
}

#[cfg(test)]
mod tests {
    use super::Conv2D;
    use crate as mu;
    use crate::tests::equal_arrays;
    use crate::Tensor;

    #[test]
    fn conv2d_forward_backward() {
        let conv2d = Conv2D::<1, 1, 1, 1, _>::new(mu::fill(1.0));
        let x = mu::fill::<1, 1, 1, 1>(0.5);

        let z = conv2d.forward(&x);
        assert!(equal_arrays(z.data(), arrayfire::constant!(0.5; 1,1,1,1)));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            arrayfire::constant!(1.0; 1, 1, 1, 1)
        ));
        assert!(equal_arrays(
            conv2d.parameters().grad().clone(),
            arrayfire::constant!(0.5; 1, 1, 1, 1)
        ));
    }

    #[test]
    fn conv2d_freeze_unfreeze() {
        let conv2d = Conv2D::<3, 5, 2, 2, _>::randn();
        let conv2d = conv2d.freeze();
        let _ = conv2d.unfreeze();
    }
}

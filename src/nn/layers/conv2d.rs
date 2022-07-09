use crate::{
    graph::node::Node,
    tensor::{
        constant::Constant,
        traits::{Data, Pair, Tensed},
        variable::Variable,
        Tensor,
    },
};
use arrayfire::{dim4, Array, ConvGradientType};
use std::rc::Rc;

/// A 2 dimensional convolutional layer with `I` input channels, `O` output channels and `H` height and `W` width kernel size
pub struct Conv2D<const I: u64, const O: u64, const H: u64, const W: u64, T: Data = Variable>(
    Tensor<O, I, H, W, T>,
);

impl<const I: u64, const O: u64, const H: u64, const W: u64, T: Data> Conv2D<I, O, H, W, T> {
    /// Given an input computes the output
    #[inline]
    pub fn forward<X: Tensed<CHANNELS = { I }>>(
        &self,
        x: &X,
    ) -> Tensor<
        { X::BATCH },
        O,
        { X::HEIGHT - H + 1 },
        { X::WIDTH - W + 1 },
        <X::Data as Pair<T>>::Output,
    >
    where
        <X as Tensed>::Data: Pair<T>,
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

impl<const I: u64, const O: u64, const H: u64, const W: u64> Conv2D<I, O, H, W, Variable> {
    /// Returns a new `Conv2D` layer with its weights and biases taken from a normal
    /// distribution with mean 0 and standard deviation 1
    #[must_use]
    #[inline]
    pub fn randn() -> Self {
        Self(crate::randn())
    }

    /// Consumes this layer and returns a copy with constant parameters
    #[must_use]
    #[inline]
    pub fn freeze(self) -> Conv2D<I, O, H, W, Constant> {
        Conv2D(self.0.freeze())
    }

    /// Returns the layer's trainable parameters
    #[must_use]
    #[inline]
    pub fn parameters(&self) -> Rc<Node> {
        self.0.inner().node()
    }
}

impl<const I: u64, const O: u64, const H: u64, const W: u64> Conv2D<I, O, H, W, Constant> {
    /// Consumes this layer and returns a copy with trainable parameters
    #[must_use]
    #[inline]
    pub fn unfreeze(self) -> Conv2D<I, O, H, W, Variable> {
        Conv2D(self.0.unfreeze())
    }
}

#[cfg(test)]
mod tests {
    use super::Conv2D;
    use crate as mu;
    use crate::tensor::traits::Tensed;
    use crate::tests::equal_data;

    #[test]
    fn conv2d_forward_backward() {
        let conv2d = Conv2D::<1, 1, 1, 1>(mu::fill(1.0));
        let x = mu::fill::<1, 1, 1, 1>(0.5);

        let z = conv2d.forward(&x);
        assert!(equal_data(z.data(), arrayfire::constant!(0.5; 1,1,1,1)));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            arrayfire::constant!(1.0; 1, 1, 1, 1)
        ));
        assert!(equal_data(
            conv2d.parameters().grad().clone(),
            arrayfire::constant!(0.5; 1, 1, 1, 1)
        ));
    }

    #[test]
    fn conv2d_freeze_unfreeze() {
        let conv2d = Conv2D::<3, 5, 2, 2>::randn();
        let conv2d = conv2d.freeze();
        let _ = conv2d.unfreeze();
    }
}

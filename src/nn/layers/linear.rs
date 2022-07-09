use crate::{
    graph::node::Node,
    tensor::{
        constant::Constant,
        traits::{Data, Pair, Tensed},
        variable::Variable,
        Tensor,
    },
};
use arrayfire::{seq, view, Array, MatProp};
use std::rc::Rc;

/// A Linear (perceptron) neural network layer with `I` input size and `O` output size
#[allow(clippy::cast_possible_truncation)]
pub struct Linear<const I: u64, const O: u64, T: Data = Variable>(Tensor<1, 1, { I + 1 }, O, T>)
where
    [(); (I + 1) as usize]:;

#[allow(clippy::cast_possible_truncation)]
impl<const I: u64, const O: u64, T: Data> Linear<I, O, T>
where
    [(); (I + 1) as usize]:,
{
    /// Given an input computes the output
    #[inline]
    pub fn forward<X: Tensed<CHANNELS = 1, HEIGHT = 1, WIDTH = { I }>>(
        &self,
        x: &X,
    ) -> Tensor<{ X::BATCH }, 1, 1, O, <X::Data as Pair<T>>::Output>
    where
        <X as Tensed>::Data: Pair<T>,
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

#[allow(clippy::cast_possible_truncation)]
impl<const I: u64, const O: u64> Linear<I, O, Variable>
where
    [(); (I + 1) as usize]:,
{
    /// Returns a new Linear layer with its weights and biases taken from a normal
    /// distribution with mean 0 and standard deviation 1
    #[must_use]
    #[inline]
    pub fn randn() -> Self {
        Self(crate::randn())
    }

    /// Consumes this layer and returns it with constant (not trainable) parameters
    #[must_use]
    #[inline]
    pub fn freeze(self) -> Linear<I, O, Constant> {
        Linear(self.0.freeze())
    }

    /// Get the layer's trainable parameters
    #[must_use]
    #[inline]
    pub fn parameters(&self) -> Rc<Node> {
        self.0.inner().node()
    }
}

#[allow(clippy::cast_possible_truncation)]
impl<const I: u64, const O: u64> Linear<I, O, Constant>
where
    [(); (I + 1) as usize]:,
{
    /// Consumes this layer and returns it with variable (trainable) parameters
    #[must_use]
    #[inline]
    pub fn unfreeze(self) -> Linear<I, O, Variable> {
        Linear(self.0.unfreeze())
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate as mu;
    use crate::tensor::traits::Tensed;
    use crate::tests::equal_data;
    use arrayfire::Array;

    #[test]
    fn linear_forward_backward() {
        let linear = Linear::<3, 5>(mu::fill(1.0));
        let x = mu::fill::<1, 1, 1, 3>(0.5);

        let z = linear.forward(&x);
        assert!(equal_data(z.data(), arrayfire::constant!(2.5; 1, 5, 1, 1)));

        z.backward();
        assert!(equal_data(
            x.grad().data(),
            arrayfire::constant!(5.0; 1, 3, 1, 1)
        ));
        assert!(equal_data(
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
        let linear = Linear::<3, 5>::randn();
        let linear = linear.freeze();
        let _ = linear.unfreeze();
    }
}

use crate as mu;
use crate::graph::node::Node;
use crate::tensor::{constant::Constant, params::DoubleParam, variable::Variable, Tensor};
use std::rc::Rc;

/// A Linear (perceptron) neural network layer with `B` batch size, `I` input size and `O` output size
pub struct Linear<const B: u64, const I: u64, const O: u64, W, P> {
    weights: W,
    bias: P,
}

impl<const B: u64, const I: u64, const O: u64, W, P> Linear<B, I, O, W, P>
where
    W: Tensor<B, 1, I, O>,
    P: Tensor<B, 1, 1, O>,
{
    /// Given an input computes the output
    #[inline]
    pub fn forward<X, Y>(&self, x: &X) -> Y
    where
        X: Tensor<B, 1, 1, I> + DoubleParam<W, Y>,
        Y: Tensor<B, 1, 1, O> + DoubleParam<P, Y>,
    {
        x.mm(&self.weights).add(&self.bias)
    }
}

impl<const B: u64, const I: u64, const O: u64>
    Linear<B, I, O, Variable<B, 1, I, O>, Variable<B, 1, 1, O>>
{
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            weights: mu::randn(),
            bias: mu::randn(),
        }
    }

    /// Consumes this layer and returns a copy with constant parameters
    #[must_use]
    #[inline]
    pub fn freeze(self) -> Linear<B, I, O, Constant<B, 1, I, O>, Constant<B, 1, 1, O>> {
        Linear {
            weights: self.weights.freeze(),
            bias: self.bias.freeze(),
        }
    }

    /// Returns the layer parameters as an array of computation graph nodes
    #[must_use]
    #[inline]
    pub fn parameters(&self) -> [Rc<Node>; 2] {
        [(&self.weights).into(), (&self.bias).into()]
    }
}

impl<const B: u64, const I: u64, const O: u64>
    Linear<B, I, O, Constant<B, 1, I, O>, Constant<B, 1, 1, O>>
{
    /// Consumes this layer and returns a copy with trainable parameters
    #[must_use]
    #[inline]
    pub fn unfreeze(self) -> Linear<B, I, O, Variable<B, 1, I, O>, Variable<B, 1, 1, O>> {
        Linear {
            weights: self.weights.unfreeze(),
            bias: self.bias.unfreeze(),
        }
    }
}

impl<const B: u64, const I: u64, const O: u64> Default
    for Linear<B, I, O, Variable<B, 1, I, O>, Variable<B, 1, 1, O>>
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate as mu;
    use crate::Tensor;

    #[test]
    fn linear_freeze_unfreeze() {
        let linear = Linear::<1, 3, 5, _, _>::new().freeze().unfreeze();

        let x = mu::fill::<1, 1, 1, 3>(2.0).freeze();
        let z = linear.forward(&x);

        assert_eq!(z.data().dims(), arrayfire::dim4!(1, 5, 1, 1));
    }
}

use crate::graph::{node::Node, tape::Tape};
use crate::tensor::{variable::Variable, Tensor};
use arrayfire::Array;

/// A non-differentiable tensor that's not tracked in the computation graph
pub struct Constant<const B: u64, const L: u64, const R: u64, const C: u64>(Array<f32>);

impl<const B: u64, const L: u64, const R: u64, const C: u64> Constant<B, L, R, C> {
    /// Creates a new constant
    pub(crate) const fn new(data: Array<f32>) -> Self {
        Self(data)
    }

    /// Consume this `Constant` tensor and return a `Variable` that is tracked in the
    /// computation graph from now on
    pub fn unfreeze(self) -> Variable<B, L, R, C> {
        Variable::new(Tape::default(), Node::declaration(self.0))
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> Tensor<B, L, R, C>
    for Constant<B, L, R, C>
{
    fn data(&self) -> Array<f32> {
        self.0.clone()
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> From<&Constant<B, L, R, C>>
    for Array<f32>
{
    #[inline]
    fn from(cons: &Constant<B, L, R, C>) -> Self {
        cons.data()
    }
}

#[cfg(test)]
mod tests {
    use super::Constant;
    use crate::tensor::Tensor;
    use crate::tests::equal_arrays;

    #[test]
    fn new_constant() {
        let tensor = Constant::<3, 4, 2, 1>::new(arrayfire::constant!(5.0; 2,1,4,3));
        assert!(equal_arrays(
            (&tensor).into(),
            arrayfire::constant!(5.0; 2,1,4,3)
        ))
    }

    #[test]
    fn constant_unfreeze() {
        let tensor = Constant::<3, 4, 2, 1>::new(arrayfire::constant!(5.0; 2,1,4,3)).unfreeze();
        assert!(equal_arrays(
            tensor.data(),
            arrayfire::constant!(5.0; 2,1,4,3)
        ));
        assert!(matches!(tensor.tape().nodes().len(), 1));
        assert!(equal_arrays(
            tensor.grad().data(),
            arrayfire::constant!(0.0; 2,1,4,3)
        ));
    }
}

use crate::graph::{node::Node, tape::Tape};
use crate::tensor::{constant::Constant, Tensor};
use arrayfire::Array;
use std::rc::Rc;

/// A differentiable tensor being tracked in the computation graph
pub struct Variable<const B: u64, const C: u64, const H: u64, const W: u64> {
    tape: Tape,
    node: Rc<Node>,
}

impl<const B: u64, const C: u64, const H: u64, const W: u64> Variable<B, C, H, W> {
    /// Creates a new variable. It assumes its node has already been pushed to the computation graph.
    pub(crate) fn new(mut tape: Tape, node: Node) -> Self {
        let node = Rc::new(node);
        tape.push(node.clone());
        Self { tape, node }
    }

    /// Returns the graph where this tensor is being tracked
    pub(crate) const fn tape(&self) -> &Tape {
        &self.tape
    }

    /// Starting from this tensor node, compute the reverse auto differentiation.
    /// Once called, all the ancestor nodes for which this tensor depends on will have
    /// their gradients filled with the derivative with respect to this tensor
    pub fn backward(&self) {
        // derivative of self wrt to self is one
        self.node.ones_grad();
        for node in self.tape.nodes().rev() {
            node.reverse();
        }
    }

    /// Set all gradients to zero, including this tensor's and all its ancestors
    pub fn reset(&self) {
        for node in self.tape.nodes().rev() {
            node.zero_grad();
        }
    }

    /// Returns as a new `Variable` tensor, the gradients of `Z` with respect to itself.
    /// Where `Z` is the tensor for which `backward` was called. Note that if `backward` was
    /// not called, or this tensor is not a dependency of `Z` the result will always be a
    /// tensor filled with zeros. If on the other hand, this is the same tensor as the one
    /// where `backward` was called, the result will always be filled with ones, because that
    /// is `dz/dz`
    pub fn grad(&self) -> Self {
        Self::new(Tape::default(), Node::declaration(self.node.grad().clone()))
    }

    /// Consume this `Variable` tensor and return a `Constant` that is not tracked in the
    /// computation graph
    pub fn freeze(self) -> Constant<B, C, H, W> {
        Constant::new(self.data())
    }
}

impl<const B: u64, const C: u64, const H: u64, const W: u64> From<&Variable<B, C, H, W>>
    for Rc<Node>
{
    #[inline]
    fn from(tensor: &Variable<B, C, H, W>) -> Self {
        tensor.node.clone()
    }
}

impl<const B: u64, const C: u64, const H: u64, const W: u64> Tensor for Variable<B, C, H, W> {
    const BATCH: u64 = B;
    const CHANNELS: u64 = C;
    const HEIGHT: u64 = H;
    const WIDTH: u64 = W;

    fn data(&self) -> Array<f32> {
        self.node.data().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Variable;
    use crate::graph::{node::Node, tape::Tape};
    use crate::tensor::Tensor;
    use crate::tests::equal_arrays;

    #[test]
    fn new_variable() {
        let tensor = Variable::<3, 4, 2, 1>::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 2,1,4,3)),
        );
        assert!(equal_arrays(
            tensor.data(),
            arrayfire::constant!(5.0; 2,1,4,3)
        ));
        assert_eq!(tensor.node.id(), 0);
    }

    #[test]
    fn variable_freeze() {
        let tensor = Variable::<3, 4, 2, 1>::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 2,1,4,3)),
        )
        .freeze();
        assert!(equal_arrays(
            (&tensor).into(),
            arrayfire::constant!(5.0; 2,1,4,3)
        ));
    }

    #[test]
    fn variable_backward() {
        let tensor = Variable::<3, 4, 2, 1>::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 2,1,4,3)),
        );
        tensor.backward();
        assert!(equal_arrays(
            tensor.grad().data(),
            arrayfire::constant!(1.0; 2,1,4,3)
        ));
    }
}

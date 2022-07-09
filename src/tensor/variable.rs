use crate::{
    graph::{
        node::{BinaryReverseFn, Node, UnaryReverseFn},
        tape::Tape,
    },
    tensor::{
        constant::Constant,
        traits::{Data, Pair},
    },
};
use arrayfire::Array;
use std::rc::Rc;

/// Data for a tensor being tracked in the computation graph
#[derive(Clone)]
pub struct Variable {
    tape: Tape,
    node: Rc<Node>,
}

impl Variable {
    /// Constructs variable data from the given tape and node. The node is pushed to the tape.
    pub fn new(mut tape: Tape, node: Node) -> Self {
        let node = Rc::new(node);
        tape.push(node.clone());
        Self { tape, node }
    }

    /// Returns the gradients of the holded data as an arrayfire array
    pub fn grad(&self) -> Array<f32> {
        self.node.grad().clone()
    }

    /// Returns the tape tracking the computation graph up until the existence of this variable
    pub const fn tape(&self) -> &Tape {
        &self.tape
    }

    /// Returns the node in the computation graph holding the data and gradients of this variable
    pub fn node(&self) -> Rc<Node> {
        self.node.clone()
    }
}

impl Data for Variable {
    fn push_unary(&self, data: Array<f32>, reverse: UnaryReverseFn, args: &[Array<f32>]) -> Self {
        let node = Node::unary(data, self.node(), reverse, args);
        Self::new(self.tape().clone(), node)
    }

    fn values(&self) -> Array<f32> {
        self.node().data().clone()
    }
}

impl Pair<Self> for Variable {
    type Output = Self;

    fn push_binary(
        &self,
        other: &Self,
        data: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Output {
        let node = Node::binary_varvar(data, (self.node(), other.node()), reverse, args);
        Self::new(self.tape().merge(other.tape()), node)
    }
}

impl Pair<Constant> for Variable {
    type Output = Self;

    fn push_binary(
        &self,
        _other: &Constant,
        data: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Output {
        let node = Node::binary_varconst(data, self.node(), reverse, args);
        Self::new(self.tape().clone(), node)
    }
}

impl From<Array<f32>> for Variable {
    fn from(data: Array<f32>) -> Self {
        Self::new(Tape::default(), Node::declaration(data))
    }
}

#[cfg(test)]
mod tests {
    use super::Variable;
    use crate::graph::{node::Node, tape::Tape};
    use crate::tensor::{
        traits::{Data, Pair},
        Constant,
    };
    use crate::tests::equal_data;

    #[test]
    fn new() {
        let variable = Variable::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 1,1,1,1)),
        );
        assert!(equal_data(
            variable.values(),
            arrayfire::constant!(5.0; 1,1,1,1)
        ))
    }

    #[test]
    fn push_unary() {
        let variable = Variable::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 1,1,1,1)),
        );
        let variable = variable.push_unary(
            arrayfire::constant!(2.0; 1,1,1,1),
            |_, _| arrayfire::constant!(1.0; 1,1,1,1),
            &[],
        );
        assert!(equal_data(
            variable.grad(),
            arrayfire::constant!(0.0; 1,1,1,1)
        ))
    }

    #[test]
    fn push_binary_constant() {
        let variable = Variable::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 1,1,1,1)),
        );
        let other = Constant::new(arrayfire::constant!(4.0; 1,1,1,1));
        let variable = variable.push_binary(
            &other,
            arrayfire::constant!(2.0; 1,1,1,1),
            |_, _| {
                (
                    arrayfire::constant!(1.0; 1,1,1,1),
                    arrayfire::constant!(1.0; 1,1,1,1),
                )
            },
            &[],
        );
        assert!(equal_data(
            variable.grad(),
            arrayfire::constant!(0.0; 1,1,1,1)
        ))
    }

    #[test]
    fn push_binary_variable() {
        let variable = Variable::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(5.0; 1,1,1,1)),
        );
        let other = Variable::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(4.0; 1,1,1,1)),
        );
        let variable = variable.push_binary(
            &other,
            arrayfire::constant!(2.0; 1,1,1,1),
            |_, _| {
                (
                    arrayfire::constant!(1.0; 1,1,1,1),
                    arrayfire::constant!(1.0; 1,1,1,1),
                )
            },
            &[],
        );
        assert!(equal_data(
            variable.grad(),
            arrayfire::constant!(0.0; 1,1,1,1)
        ))
    }
}

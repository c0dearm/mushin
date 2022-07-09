use crate::{
    graph::node::{BinaryReverseFn, Node, UnaryReverseFn},
    tensor::{
        traits::{Data, Pair},
        variable::Variable,
    },
};
use arrayfire::Array;

/// Data for a non-differentiable tensor not tracked in the computation graph
#[derive(Clone)]
pub struct Constant(Array<f32>);

impl Constant {
    /// Constructs constant data from a given array
    pub fn new(data: Array<f32>) -> Self {
        Self(data)
    }
}

impl Data for Constant {
    fn push_unary(&self, data: Array<f32>, _reverse: UnaryReverseFn, _args: &[Array<f32>]) -> Self {
        Self::new(data)
    }

    fn values(&self) -> Array<f32> {
        self.0.clone()
    }
}

impl Pair<Variable> for Constant {
    type Output = Variable;

    fn push_binary(
        &self,
        other: &Variable,
        data: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Output {
        let node = Node::binary_constvar(data, other.node(), reverse, args);
        Variable::new(other.tape().clone(), node)
    }
}

impl Pair<Self> for Constant {
    type Output = Self;

    fn push_binary(
        &self,
        _other: &Self,
        data: Array<f32>,
        _reverse: BinaryReverseFn,
        _args: &[Array<f32>],
    ) -> Self::Output {
        Self::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::Constant;
    use crate::graph::{node::Node, tape::Tape};
    use crate::tensor::{
        traits::{Data, Pair},
        Variable,
    };
    use crate::tests::equal_data;

    #[test]
    fn new() {
        let constant = Constant::new(arrayfire::constant!(5.0; 1,1,1,1));
        assert!(equal_data(
            constant.values(),
            arrayfire::constant!(5.0; 1,1,1,1)
        ))
    }

    #[test]
    fn push_unary() {
        let constant = Constant::new(arrayfire::constant!(5.0; 1,1,1,1));
        let constant = constant.push_unary(
            arrayfire::constant!(2.0; 1,1,1,1),
            |_, _| arrayfire::constant!(1.0; 1,1,1,1),
            &[],
        );
        assert!(equal_data(
            constant.values(),
            arrayfire::constant!(2.0; 1,1,1,1)
        ))
    }

    #[test]
    fn push_binary_constant() {
        let constant = Constant::new(arrayfire::constant!(5.0; 1,1,1,1));
        let other = Constant::new(arrayfire::constant!(4.0; 1,1,1,1));
        let constant = constant.push_binary(
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
            constant.values(),
            arrayfire::constant!(2.0; 1,1,1,1)
        ))
    }

    #[test]
    fn push_binary_variable() {
        let constant = Constant::new(arrayfire::constant!(5.0; 1,1,1,1));
        let other = Variable::new(
            Tape::default(),
            Node::declaration(arrayfire::constant!(4.0; 1,1,1,1)),
        );
        let variable = constant.push_binary(
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

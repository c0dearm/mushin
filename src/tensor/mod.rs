//! This module includes the `Tensor` type, which holds either `Variable`
//! or `Constant` data.
//!
//! `Variable` tensors are tracked in the computation graph, they are
//! differentiable and include methods like `backward` and `grad` to
//! respectively compute and retrieve the gradients.
//!
//! `Constant` tensors are not tracked so they are not differentiable and
//! do not have the `backward` and `grad` methods.
//!
//! `Variable` and `Constant` tensors are interoperable in the sense that you
//! can perform any operation between them no matter the combination of types.
//! Only operations between constants will result in a `Constant` tensor, otherwise
//! a `Variable` tensor is returned.
//!
//! At any time a `Variable` tensor can be frozen by calling the `freeze` method,
//! which will consume it and return a `Constant`. In the same fashion, a `Constant`
//! can be unfrozen by calling the `unfreeze` method, which will return a Variable
//! tracked in the computation graph.

pub mod constant;
pub mod traits;
pub mod variable;

use crate::graph::{
    node::{BinaryReverseFn, Node, UnaryReverseFn},
    tape::Tape,
};
use arrayfire::Array;
use constant::Constant;
use traits::{Data, Pair, Tensed};
use variable::Variable;

#[derive(Clone)]
pub struct Tensor<const B: u64, const C: u64, const H: u64, const W: u64, D: Data>(D);

impl<const B: u64, const C: u64, const H: u64, const W: u64> Tensor<B, C, H, W, Variable> {
    /// Returns the tensor gradients as another variable tensor
    pub fn grad(&self) -> Self {
        Self(Variable::new(
            Tape::default(),
            Node::declaration(self.0.grad()),
        ))
    }

    /// Consumes the variable tensor and returns it as a constant tensor
    pub fn freeze(self) -> Tensor<B, C, H, W, Constant> {
        Tensor(Constant::new(self.data()))
    }

    /// Starting from this tensor node, compute the reverse auto differentiation.
    /// Once called, all the ancestor nodes for which this tensor depends on will have
    /// their gradients filled with the derivative with respect to this tensor
    pub fn backward(&self) {
        // derivative of self wrt to self is one
        self.0.node().ones_grad();
        for node in self.0.tape().nodes().rev() {
            node.reverse();
        }
    }

    /// Set all gradients to zero, including this tensor's and all its ancestors
    pub fn reset(&self) {
        for node in self.0.tape().nodes().rev() {
            node.zero_grad();
        }
    }
}

impl<const B: u64, const C: u64, const H: u64, const W: u64> Tensor<B, C, H, W, Constant> {
    /// Consumes the constant tensor and returns it as a variable tensor
    pub fn unfreeze(self) -> Tensor<B, C, H, W, Variable> {
        Tensor(Variable::new(
            Tape::default(),
            Node::declaration(self.data()),
        ))
    }
}

impl<const B: u64, const C: u64, const H: u64, const W: u64, D: Data> Tensed
    for Tensor<B, C, H, W, D>
{
    type Data = D;
    const BATCH: u64 = B;
    const CHANNELS: u64 = C;
    const HEIGHT: u64 = H;
    const WIDTH: u64 = W;

    fn inner(&self) -> &Self::Data {
        &self.0
    }

    fn push_unary<const YB: u64, const YC: u64, const YH: u64, const YW: u64>(
        &self,
        data: Array<f32>,
        reverse: UnaryReverseFn,
        args: &[Array<f32>],
    ) -> Tensor<YB, YC, YH, YW, D> {
        Tensor(self.0.push_unary(data, reverse, args))
    }

    fn push_binary<const ZB: u64, const ZC: u64, const ZH: u64, const ZW: u64, Y: Tensed>(
        &self,
        other: &Y,
        data: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Tensor<ZB, ZC, ZH, ZW, <Self::Data as Pair<Y::Data>>::Output>
    where
        Self::Data: Pair<Y::Data>,
    {
        Tensor(self.0.push_binary(other.inner(), data, reverse, args))
    }
}

impl<const B: u64, const C: u64, const H: u64, const W: u64> From<Constant>
    for Tensor<B, C, H, W, Constant>
{
    fn from(constant: Constant) -> Self {
        Self(constant)
    }
}

impl<const B: u64, const C: u64, const H: u64, const W: u64> From<Variable>
    for Tensor<B, C, H, W, Variable>
{
    fn from(variable: Variable) -> Self {
        Self(variable)
    }
}

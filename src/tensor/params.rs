//! This module implements a pair of traits for `Variable` and `Constant` tensors
//! that are used at compile time to determine if the resulting tensor in a given
//! operation is going to be a `Variable` (tracked in the computation graph) or a
//! `Constant` (not tracked in the computation graph). As long as one of the
//! parameters is a `Variable` the resulting tensor is also a `Variable`.
//! Check the following tables for a complete overview:
//!
//! ## Single parameter operation
//! | Parameter |  Output  |
//! |-----------|----------|
//! | Variable  | Variable |
//! | Constant  | Constant |
//!
//! ## Double parameter operation
//! | First parameter | Second parameter |  Output  |
//! |-----------------|------------------|----------|
//! |     Variable    |      Variable    | Variable |
//! |     Variable    |      Constant    | Variable |
//! |     Constant    |      Variable    | Variable |
//! |     Constant    |      Constant    | Constant |

use arrayfire::Array;

use crate::graph::node::{BinaryReverseFn, Node, UnaryReverseFn};
use crate::tensor::{constant::Constant, variable::Variable};

/// Determines the output of single parameter operations (unary)
pub trait SingleParam<const YB: u64, const YC: u64, const YH: u64, const YW: u64> {
    type Out;

    /// Creates a new tensor with the given result as data and pushes it to the computation graph (if required)
    fn push_unary(
        &self,
        result: Array<f32>,
        reverse: UnaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Out;
}

impl<
        const XB: u64,
        const XC: u64,
        const XH: u64,
        const XW: u64,
        const YB: u64,
        const YC: u64,
        const YH: u64,
        const YW: u64,
    > SingleParam<YB, YC, YH, YW> for Variable<XB, XC, XH, XW>
{
    type Out = Variable<YB, YC, YH, YW>;

    fn push_unary(
        &self,
        result: Array<f32>,
        reverse: UnaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Out {
        let node = Node::unary(result, self.into(), reverse, args);
        Variable::new(self.tape().clone(), node)
    }
}

impl<
        const XB: u64,
        const XC: u64,
        const XH: u64,
        const XW: u64,
        const YB: u64,
        const YC: u64,
        const YH: u64,
        const YW: u64,
    > SingleParam<YB, YC, YH, YW> for Constant<XB, XC, XH, XW>
{
    type Out = Constant<YB, YC, YH, YW>;

    fn push_unary(
        &self,
        result: Array<f32>,
        _reverse: UnaryReverseFn,
        _args: &[Array<f32>],
    ) -> Self::Out {
        Constant::new(result)
    }
}

/// Determines the output of double parameter operations (binary)
pub trait DoubleParam<const ZB: u64, const ZC: u64, const ZH: u64, const ZW: u64, Y> {
    type Out;

    /// Creates a new tensor with the given result as data and pushes it to the computation graph (if required)
    fn push_binary(
        &self,
        other: &Y,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Out;
}

impl<
        const XB: u64,
        const XC: u64,
        const XH: u64,
        const XW: u64,
        const YB: u64,
        const YC: u64,
        const YH: u64,
        const YW: u64,
        const ZB: u64,
        const ZC: u64,
        const ZH: u64,
        const ZW: u64,
    > DoubleParam<ZB, ZC, ZH, ZW, Variable<YB, YC, YH, YW>> for Variable<XB, XC, XH, XW>
{
    type Out = Variable<ZB, ZC, ZH, ZW>;

    fn push_binary(
        &self,
        other: &Variable<YB, YC, YH, YW>,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Out {
        let node = Node::binary_varvar(result, (self.into(), other.into()), reverse, args);
        Variable::new(self.tape().merge(other.tape()), node)
    }
}

impl<
        const XB: u64,
        const XC: u64,
        const XH: u64,
        const XW: u64,
        const YB: u64,
        const YC: u64,
        const YH: u64,
        const YW: u64,
        const ZB: u64,
        const ZC: u64,
        const ZH: u64,
        const ZW: u64,
    > DoubleParam<ZB, ZC, ZH, ZW, Constant<YB, YC, YH, YW>> for Variable<XB, XC, XH, XW>
{
    type Out = Variable<ZB, ZC, ZH, ZW>;

    fn push_binary(
        &self,
        _other: &Constant<YB, YC, YH, YW>,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Out {
        let node = Node::binary_varconst(result, self.into(), reverse, args);
        Variable::new(self.tape().merge(self.tape()), node)
    }
}

impl<
        const XB: u64,
        const XC: u64,
        const XH: u64,
        const XW: u64,
        const YB: u64,
        const YC: u64,
        const YH: u64,
        const YW: u64,
        const ZB: u64,
        const ZC: u64,
        const ZH: u64,
        const ZW: u64,
    > DoubleParam<ZB, ZC, ZH, ZW, Variable<YB, YC, YH, YW>> for Constant<XB, XC, XH, XW>
{
    type Out = Variable<ZB, ZC, ZH, ZW>;

    fn push_binary(
        &self,
        other: &Variable<YB, YC, YH, YW>,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Out {
        let node = Node::binary_constvar(result, other.into(), reverse, args);
        Variable::new(other.tape().merge(other.tape()), node)
    }
}

impl<
        const XB: u64,
        const XC: u64,
        const XH: u64,
        const XW: u64,
        const YB: u64,
        const YC: u64,
        const YH: u64,
        const YW: u64,
        const ZB: u64,
        const ZC: u64,
        const ZH: u64,
        const ZW: u64,
    > DoubleParam<ZB, ZC, ZH, ZW, Constant<YB, YC, YH, YW>> for Constant<XB, XC, XH, XW>
{
    type Out = Constant<ZB, ZC, ZH, ZW>;

    fn push_binary(
        &self,
        _other: &Constant<YB, YC, YH, YW>,
        result: Array<f32>,
        _reverse: BinaryReverseFn,
        _args: &[Array<f32>],
    ) -> Self::Out {
        Constant::new(result)
    }
}

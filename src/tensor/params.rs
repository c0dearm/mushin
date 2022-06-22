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
pub trait SingleParam<Y> {
    /// Creates a new tensor with the given result as data and pushes it to the computation graph (if required)
    fn push_unary(&self, result: Array<f32>, reverse: UnaryReverseFn, args: &[Array<f32>]) -> Y;
}

impl<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
    > SingleParam<Variable<YB, YL, YR, YC>> for Variable<B, L, R, C>
{
    fn push_unary(
        &self,
        result: Array<f32>,
        reverse: UnaryReverseFn,
        args: &[Array<f32>],
    ) -> Variable<YB, YL, YR, YC> {
        let node = Node::unary(result, self.into(), reverse, args);
        Variable::new(self.tape().clone(), node)
    }
}

impl<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
    > SingleParam<Constant<YB, YL, YR, YC>> for Constant<B, L, R, C>
{
    fn push_unary(
        &self,
        result: Array<f32>,
        _reverse: UnaryReverseFn,
        _args: &[Array<f32>],
    ) -> Constant<YB, YL, YR, YC> {
        Constant::new(result)
    }
}

/// Determines the output of double parameter operations (binary)
pub trait DoubleParam<Y, Z> {
    /// Creates a new tensor with the given result as data and pushes it to the computation graph (if required)
    fn push_binary(
        &self,
        other: &Y,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Z;
}

impl<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > DoubleParam<Variable<YB, YL, YR, YC>, Variable<ZB, ZL, ZR, ZC>> for Variable<B, L, R, C>
{
    fn push_binary(
        &self,
        other: &Variable<YB, YL, YR, YC>,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Variable<ZB, ZL, ZR, ZC> {
        let node = Node::binary_varvar(result, (self.into(), other.into()), reverse, args);
        Variable::new(self.tape().merge(other.tape()), node)
    }
}

impl<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > DoubleParam<Constant<YB, YL, YR, YC>, Variable<ZB, ZL, ZR, ZC>> for Variable<B, L, R, C>
{
    fn push_binary(
        &self,
        _other: &Constant<YB, YL, YR, YC>,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Variable<ZB, ZL, ZR, ZC> {
        let node = Node::binary_varconst(result, self.into(), reverse, args);
        Variable::new(self.tape().clone(), node)
    }
}

impl<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > DoubleParam<Variable<YB, YL, YR, YC>, Variable<ZB, ZL, ZR, ZC>> for Constant<B, L, R, C>
{
    fn push_binary(
        &self,
        other: &Variable<YB, YL, YR, YC>,
        result: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Variable<ZB, ZL, ZR, ZC> {
        let node = Node::binary_constvar(result, other.into(), reverse, args);
        Variable::new(other.tape().clone(), node)
    }
}

impl<
        const B: u64,
        const L: u64,
        const R: u64,
        const C: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > DoubleParam<Constant<YB, YL, YR, YC>, Constant<ZB, ZL, ZR, ZC>> for Constant<B, L, R, C>
{
    fn push_binary(
        &self,
        _other: &Constant<YB, YL, YR, YC>,
        result: Array<f32>,
        _reverse: BinaryReverseFn,
        _args: &[Array<f32>],
    ) -> Constant<ZB, ZL, ZR, ZC> {
        Constant::new(result)
    }
}

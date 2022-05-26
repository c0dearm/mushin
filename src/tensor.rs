use arrayfire::{constant, dim4, identity, randn, randu, Array};

use crate::context::{
    function::{
        ConstantParameter, DoubleParamReverseFn, DoubleParameter, SingleParamReverseFn,
        VariableParameter,
    },
    tape::{NodeId, Tape},
};

/// Possible pre-defined values to create a tensor from
#[non_exhaustive]
pub enum Values {
    /// The identity tensor
    Identity,
    /// Values come from a uniform distribution
    Uniform,
    /// Values come from a normal distribution
    Normal,
    /// All values zero except for the main diagonal
    Eye(f32),
    /// All values set to the given value
    Fill(f32),
    /// Custom values
    Custom(Array<f32>),
}

impl From<Array<f32>> for Values {
    #[inline]
    fn from(a: Array<f32>) -> Self {
        Self::Custom(a)
    }
}

/// A tensor which is considered a constant (doesn't contribute to the computation graph)
pub struct Constant<const B: u64, const L: u64, const R: u64, const C: u64> {
    /// Holds the tensor values
    value: Array<f32>,
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> Constant<B, L, R, C> {
    pub fn new(values: Values) -> Self {
        Self {
            value: Self::gen_value(values),
        }
    }
}

/// A tensor which is considered a variable (contributes to the computation graph)
pub struct Variable<'t, const B: u64, const L: u64, const R: u64, const C: u64> {
    /// Holds the tensor values
    value: Array<f32>,
    /// The computation graph
    tape: &'t Tape,
    /// Index to the node in the computation graph
    index: NodeId,
}

impl<'t, const B: u64, const L: u64, const R: u64, const C: u64> Variable<'t, B, L, R, C> {
    pub fn new(values: Values, tape: &'t Tape) -> Self {
        Self {
            value: Self::gen_value(values),
            tape,
            index: tape.push_nary(),
        }
    }

    pub const fn tape(&self) -> &Tape {
        self.tape
    }

    pub const fn index(&self) -> NodeId {
        self.index
    }
}

/// Represents a pair of tensors. Used to implement the `BinaryOp` trait on it
pub struct Pair<'a, X, Y>(pub &'a X, pub &'a Y);

/// Common trait for both `Constant` and `Variable` tensors
pub trait Tensor<const B: u64, const L: u64, const R: u64, const C: u64> {
    fn value(&self) -> &Array<f32>;

    /// Return an `Array` with the same shape as the `Tensor` filled with the provided `Values`
    fn gen_value(values: Values) -> Array<f32> {
        match values {
            Values::Identity => identity(dim4!(R, C, L, B)),
            Values::Uniform => randu!(R, C, L, B),
            Values::Normal => randn!(R, C, L, B),
            Values::Eye(x) => identity::<f32>(dim4!(R, C, L, B)) * x,
            Values::Fill(x) => constant!(x; R, C, L, B),
            Values::Custom(x) => x,
        }
    }
}

/// Trait to apply a forward function on a tensor value and push the reverse function into the computation graph
pub trait UnaryOp<Y> {
    fn eval<F: Fn(&Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        reverse: SingleParamReverseFn,
    ) -> Y;
}

/// Trait to apply a forward function on a pair of tensor values and push the reverse function into the computation graph
pub trait BinaryOp<Z> {
    fn eval<F: Fn(&Array<f32>, &Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        reverse: DoubleParamReverseFn,
    ) -> Z;
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> Tensor<B, L, R, C>
    for Constant<B, L, R, C>
{
    fn value(&self) -> &Array<f32> {
        &self.value
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> Tensor<B, L, R, C>
    for Variable<'_, B, L, R, C>
{
    fn value(&self) -> &Array<f32> {
        &self.value
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> From<&Variable<'_, B, L, R, C>>
    for VariableParameter
{
    fn from(x: &Variable<'_, B, L, R, C>) -> Self {
        Self::new(x.value.clone(), x.index)
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> From<&Constant<B, L, R, C>>
    for ConstantParameter
{
    fn from(x: &Constant<B, L, R, C>) -> Self {
        Self::new(x.value.clone())
    }
}

impl<
        const XB: u64,
        const XL: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
    > UnaryOp<Constant<YB, YL, YR, YC>> for Constant<XB, XL, XR, XC>
{
    fn eval<F: Fn(&Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        _reverse: SingleParamReverseFn,
    ) -> Constant<YB, YL, YR, YC> {
        Constant {
            value: forward(&self.value),
        }
    }
}

impl<
        't,
        const XB: u64,
        const XL: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
    > UnaryOp<Variable<'t, YB, YL, YR, YC>> for Variable<'t, XB, XL, XR, XC>
{
    fn eval<F: Fn(&Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        reverse: SingleParamReverseFn,
    ) -> Variable<'t, YB, YL, YR, YC> {
        Variable {
            value: forward(&self.value),
            tape: self.tape,
            index: self.tape.push_unary(self.into(), reverse),
        }
    }
}

impl<
        const XB: u64,
        const XL: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > BinaryOp<Constant<ZB, ZL, ZR, ZC>>
    for Pair<'_, Constant<XB, XL, XR, XC>, Constant<YB, YL, YR, YC>>
{
    fn eval<F: Fn(&Array<f32>, &Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        _reverse: DoubleParamReverseFn,
    ) -> Constant<ZB, ZL, ZR, ZC> {
        Constant {
            value: forward(&self.0.value, &self.1.value),
        }
    }
}

impl<
        't,
        const XB: u64,
        const XL: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > BinaryOp<Variable<'t, ZB, ZL, ZR, ZC>>
    for Pair<'_, Constant<XB, XL, XR, XC>, Variable<'t, YB, YL, YR, YC>>
{
    fn eval<F: Fn(&Array<f32>, &Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        reverse: DoubleParamReverseFn,
    ) -> Variable<'t, ZB, ZL, ZR, ZC> {
        Variable {
            value: forward(&self.0.value, &self.1.value),
            tape: self.1.tape,
            index: self.1.tape.push_binary(
                DoubleParameter::ConstantVariable(self.0.into(), self.1.into()),
                reverse,
            ),
        }
    }
}

impl<
        't,
        const XB: u64,
        const XL: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > BinaryOp<Variable<'t, ZB, ZL, ZR, ZC>>
    for Pair<'_, Variable<'t, XB, XL, XR, XC>, Constant<YB, YL, YR, YC>>
{
    fn eval<F: Fn(&Array<f32>, &Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        reverse: DoubleParamReverseFn,
    ) -> Variable<'t, ZB, ZL, ZR, ZC> {
        Variable {
            value: forward(&self.0.value, &self.1.value),
            tape: self.0.tape,
            index: self.0.tape.push_binary(
                DoubleParameter::VariableConstant(self.0.into(), self.1.into()),
                reverse,
            ),
        }
    }
}

impl<
        't,
        const XB: u64,
        const XL: u64,
        const XR: u64,
        const XC: u64,
        const YB: u64,
        const YL: u64,
        const YR: u64,
        const YC: u64,
        const ZB: u64,
        const ZL: u64,
        const ZR: u64,
        const ZC: u64,
    > BinaryOp<Variable<'t, ZB, ZL, ZR, ZC>>
    for Pair<'_, Variable<'t, XB, XL, XR, XC>, Variable<'t, YB, YL, YR, YC>>
{
    fn eval<F: Fn(&Array<f32>, &Array<f32>) -> Array<f32>>(
        &self,
        forward: F,
        reverse: DoubleParamReverseFn,
    ) -> Variable<'t, ZB, ZL, ZR, ZC> {
        Variable {
            value: forward(&self.0.value, &self.1.value),
            tape: self.0.tape,
            index: self.0.tape.push_binary(
                DoubleParameter::VariableVariable(self.0.into(), self.1.into()),
                reverse,
            ),
        }
    }
}

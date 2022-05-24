use arrayfire::Array;

use crate::Context;

/// Origin of a tensor. Either it came from a function (a variable) or it is a constant
#[non_exhaustive]
pub enum Origin {
    /// The value is a constant and originated from a constant function (not inserted in the tape)
    None,
    /// The value is a variable and originated from a function with given index in the tape
    Function(usize),
}
/// Possible pre-defined values to create a tensor from
#[non_exhaustive]
pub enum Values {
    /// The identity tensor
    Identity,
    /// Values come from a uniform distribution
    Uniform,
    /// Values come from a normal distribution
    Normal,
    /// Tensor with all values zero except for the main diagonal
    Eye(f32),
    /// Tensor with all values set to the given value
    Fill(f32),
}

/// The class of a tensor defines if its value matters for the computation graph
#[non_exhaustive]
#[derive(Clone, Copy)]
pub enum Class {
    /// Tensor is a constant so the value is not added to the computation graph (constants don't compute derivatives)
    Constant,
    /// Tensor is a variable and the value is added to the computation graph, the value does not persist through different builds
    Variable,
    /// Tensor is a variable and the value is added to the computation graph, that does persist through different builds.
    /// The given string is a key to retrieve the value from the persistent storage.
    Persistent(&'static str),
}

/// A mathematical tensor with a reference to its origin in the computation graph
pub struct Tensor<'ctx, const B: u64, const L: u64, const R: u64, const C: u64> {
    value: Array<f32>,
    context: &'ctx Context,
    origin: Origin,
}

impl<'ctx, const B: u64, const L: u64, const R: u64, const C: u64> Tensor<'ctx, B, L, R, C> {
    pub(crate) fn new(value: Array<f32>, origin: Origin, context: &'ctx Context) -> Self {
        Tensor {
            value,
            context,
            origin,
        }
    }

    pub(crate) const fn context(&self) -> &Context {
        self.context
    }

    pub(crate) const fn origin(&self) -> &Origin {
        &self.origin
    }
}

impl<'tsr, const B: u64, const L: u64, const R: u64, const C: u64>
    From<&'tsr Tensor<'_, B, L, R, C>> for &'tsr Array<f32>
{
    #[inline]
    fn from(t: &'tsr Tensor<'_, B, L, R, C>) -> Self {
        &t.value
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> From<&Tensor<'_, B, L, R, C>>
    for Array<f32>
{
    #[inline]
    fn from(t: &Tensor<'_, B, L, R, C>) -> Self {
        t.value.clone()
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> From<Tensor<'_, B, L, R, C>>
    for Array<f32>
{
    #[inline]
    fn from(t: Tensor<'_, B, L, R, C>) -> Self {
        t.value
    }
}

impl<const B: u64, const L: u64, const R: u64, const C: u64> TryFrom<&Tensor<'_, B, L, R, C>>
    for usize
{
    type Error = ();

    #[inline]
    fn try_from(t: &Tensor<B, L, R, C>) -> Result<Self, Self::Error> {
        match t.origin {
            Origin::Function(function) => Ok(function),
            Origin::None => Err(()),
        }
    }
}

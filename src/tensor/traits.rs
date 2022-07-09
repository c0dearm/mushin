use crate::{
    graph::node::{BinaryReverseFn, UnaryReverseFn},
    tensor::Tensor,
};
use arrayfire::Array;

/// Common methods for types holding data for a tensor. Either `Variable` or `Constant` data.
pub trait Data {
    /// Returns the tensor data as an arrayfire array
    fn values(&self) -> Array<f32>;
    /// Pushes new data, resulting from a unary operation, to the computation graph (if data is variable)
    fn push_unary(&self, data: Array<f32>, reverse: UnaryReverseFn, args: &[Array<f32>]) -> Self;
}

/// Common methods for pairs of types holding data for tensors. Depending on the combination of types,
/// the resulting data type for binary operations is either `Variable` or `Constant` data.
pub trait Pair<Y: Data> {
    /// | First parameter (Self) | Second parameter (Y) |  Output  |
    /// |------------------------|----------------------|----------|
    /// |        Variable        |        Variable      | Variable |
    /// |        Variable        |        Constant      | Variable |
    /// |        Constant        |        Variable      | Variable |
    /// |        Constant        |        Constant      | Constant |
    type Output: Data;

    /// Pushes new data, resulting from a binary operation, to the computation graph (if output is variable)
    fn push_binary(
        &self,
        other: &Y,
        data: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Self::Output;
}

/// Trait implemented for the `Tensor` type, holding either `Variable` or `Constant` data.
pub trait Tensed {
    /// `Constant` or `Variable`
    type Data: Data;

    const BATCH: u64;
    const CHANNELS: u64;
    const HEIGHT: u64;
    const WIDTH: u64;

    /// Returns the object holding the data for the tensor. Either `Variable` or `Constant`.
    fn inner(&self) -> &Self::Data;

    /// Pushes new data, resulting from a unary operation, to the computation graph (if data is variable)
    fn push_unary<const B: u64, const C: u64, const H: u64, const W: u64>(
        &self,
        data: Array<f32>,
        reverse: UnaryReverseFn,
        args: &[Array<f32>],
    ) -> Tensor<B, C, H, W, Self::Data>;

    /// Pushes new data, resulting from a binary operation, to the computation graph (if output is variable)
    fn push_binary<const B: u64, const C: u64, const H: u64, const W: u64, Y: Tensed>(
        &self,
        other: &Y,
        data: Array<f32>,
        reverse: BinaryReverseFn,
        args: &[Array<f32>],
    ) -> Tensor<B, C, H, W, <Self::Data as Pair<Y::Data>>::Output>
    where
        Self::Data: Pair<Y::Data>;

    /// Returns the tensor data as an arrayfire array
    fn data(&self) -> Array<f32> {
        self.inner().values()
    }
}

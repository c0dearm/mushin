//! This module includes the `Tensor` trait, which is implemeneted for the
//! `Variable` and `Constant` types.
//!
//! `Variable` tensors are tracked in the computation graph, so they are
//! differentiable and include methods like `backward` and `grad` to
//! respectively compute and retrieve the gradients.
//!
//! `Constant` tensors are not tracked so they are not differentiable and
//! do not have the `backward` and `grad` methods.
//!
//! `Variable` and `Constant` types are interoperable in the sense that you
//! can perform any operation between them no matter the combination of types.
//! Check the `params` module for a description on how the combination of the
//! operands influence the type of the output.
//!
//! At any time a `Variable` tensor can be frozen by calling the `freeze` method,
//! which will consume it and return a `Constant`. In the same fashion, a `Constant`
//! can be unfrozen by calling the `unfreeze` method, which will return a Variable
//! tracked in the computation graph.

pub mod constant;
pub mod params;
pub mod variable;

use crate::tensor::params::{DoubleParam, SingleParam};
use arrayfire::{constant, Array};

/// Defines operations on tensors, either `Constant` or `Variable`
pub trait Tensor<const B: u64, const L: u64, const R: u64, const C: u64>: Sized {
    /// Returns the tensor data as an `arrayfire` `Array`
    fn data(&self) -> Array<f32>;

    /// Does nothing
    #[must_use]
    #[inline]
    fn identity(&self) -> Self
    where
        Self: SingleParam<Self>,
    {
        let reverse = |df: &Array<f32>, _: &Array<f32>| df.clone();
        self.push_unary(self.data(), reverse)
    }

    /// Changes the shape of the tensor to the given dimensions
    #[inline]
    fn reshape<
        const BY: u64,
        const LY: u64,
        const RY: u64,
        const CY: u64,
        Y: Tensor<BY, LY, RY, CY>,
    >(
        &self,
    ) -> Y
    where
        Self: SingleParam<Y>,
    {
        let reverse =
            |df: &Array<f32>, _: &Array<f32>| arrayfire::moddims(df, arrayfire::dim4!(R, C, L, B));
        self.push_unary(
            arrayfire::moddims(&self.data(), arrayfire::dim4!(RY, CY, LY, BY)),
            reverse,
        )
    }

    /// Computes `sin(x)`
    #[must_use]
    #[inline]
    fn sin(&self) -> Self
    where
        Self: SingleParam<Self>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>| df * arrayfire::cos(x);
        self.push_unary(arrayfire::sin(&self.data()), reverse)
    }

    /// Computes `cos(x)`
    #[must_use]
    #[inline]
    fn cos(&self) -> Self
    where
        Self: SingleParam<Self>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>| df * -arrayfire::sin(x);
        self.push_unary(arrayfire::cos(&self.data()), reverse)
    }

    /// Sums all the elements and returns the result a single value tensor
    #[inline]
    fn sum<Y: Tensor<1, 1, 1, 1>>(&self) -> Y
    where
        Self: SingleParam<Y>,
    {
        let reverse = |df: &Array<f32>, _: &Array<f32>| df.clone();
        self.push_unary(
            constant!(arrayfire::sum_all(&self.data()).0; 1,1,1,1),
            reverse,
        )
    }

    /// Perform the element-wise addition of two tensors
    #[inline]
    fn add<Y: Tensor<B, L, R, C>, Z: Tensor<B, L, R, C>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, _: &Array<f32>, _: &Array<f32>| (df.clone(), df.clone());
        self.push_binary(
            other,
            arrayfire::add(&self.data(), &other.data(), false),
            reverse,
        )
    }

    /// Perform the element-wise substraction of two tensors
    #[inline]
    fn sub<Y: Tensor<B, L, R, C>, Z: Tensor<B, L, R, C>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, _: &Array<f32>, _: &Array<f32>| (df.clone(), -df.clone());
        self.push_binary(
            other,
            arrayfire::sub(&self.data(), &other.data(), false),
            reverse,
        )
    }

    /// Perform the element-wise multiplication of two tensors
    #[inline]
    fn mul<Y: Tensor<B, L, R, C>, Z: Tensor<B, L, R, C>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>, y: &Array<f32>| (df * y, df * x);
        self.push_binary(
            other,
            arrayfire::mul(&self.data(), &other.data(), false),
            reverse,
        )
    }

    /// Perform the element-wise division of two tensors
    #[inline]
    fn div<Y: Tensor<B, L, R, C>, Z: Tensor<B, L, R, C>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>, y: &Array<f32>| (df / y, -(df * x / y / y));
        self.push_binary(
            other,
            arrayfire::div(&self.data(), &other.data(), false),
            reverse,
        )
    }

    /// Perform the normal matrix multiplication of two tensors
    #[inline]
    fn mm<const CY: u64, Y: Tensor<B, L, C, CY>, Z: Tensor<B, L, R, CY>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>, y: &Array<f32>| {
            (
                arrayfire::matmul(df, y, arrayfire::MatProp::NONE, arrayfire::MatProp::TRANS),
                arrayfire::matmul(x, df, arrayfire::MatProp::TRANS, arrayfire::MatProp::NONE),
            )
        };
        self.push_binary(
            other,
            arrayfire::matmul(
                &self.data(),
                &other.data(),
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::NONE,
            ),
            reverse,
        )
    }

    /// Perform the element-wise power of two tensors
    #[inline]
    fn pow<Y: Tensor<B, L, R, C>, Z: Tensor<B, L, R, C>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>, y: &Array<f32>| {
            (
                df * y * arrayfire::pow(x, &(y - 1.0f32), false),
                df * arrayfire::pow(x, y, false) * arrayfire::log(x),
            )
        };
        self.push_binary(
            other,
            arrayfire::pow(&self.data(), &other.data(), false),
            reverse,
        )
    }

    /// Performs the element-wise comparison of two tensors and returns the greater values
    #[inline]
    fn maximum<Y: Tensor<B, L, R, C>, Z: Tensor<B, L, R, C>>(&self, other: &Y) -> Z
    where
        Self: DoubleParam<Y, Z>,
    {
        let reverse = |df: &Array<f32>, x: &Array<f32>, y: &Array<f32>| {
            (
                df * arrayfire::gt(x, y, false),
                df * arrayfire::gt(y, x, false),
            )
        };
        self.push_binary(
            other,
            arrayfire::maxof(&self.data(), &other.data(), false),
            reverse,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate as mu;
    use crate::{tensor::Tensor, tests::equal_arrays};
    use arrayfire::{constant, dim4, Array};

    // All result comparisons are taken from performing the exact same operations on Tensorflow

    #[test]
    fn identity_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let z = x.identity();
        assert!(equal_arrays(
            z.data(),
            Array::new(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dim4!(3, 2, 1, 1))
        ));
    }

    #[test]
    fn reshape_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let z = x.reshape::<1, 1, 1, 6, _>();
        assert!(equal_arrays(
            z.data(),
            Array::new(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0], dim4!(1, 6, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dim4!(3, 2, 1, 1))
        ));
    }

    #[test]
    fn sin_forward_backward() {
        let x = mu::eye::<1, 1, 2, 3>(0.5);
        let z = x.sin();
        assert!(equal_arrays(
            z.data(),
            Array::new(
                &[0.479425538604203, 0.0, 0.0, 0.479425538604203, 0.0, 0.0],
                dim4!(2, 3, 1, 1),
            ),
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(
                &[0.8775825618903728, 1.0, 1.0, 0.8775825618903728, 1.0, 1.0],
                dim4!(2, 3, 1, 1),
            ),
        ))
    }

    #[test]
    fn cos_forward_backward() {
        let x = mu::eye::<1, 1, 2, 3>(0.5);
        let z = x.cos();
        assert!(equal_arrays(
            z.data(),
            Array::new(
                &[0.8775825618903728, 1.0, 1.0, 0.8775825618903728, 1.0, 1.0],
                dim4!(2, 3, 1, 1),
            ),
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(
                &[-0.479425538604203, 0.0, 0.0, -0.479425538604203, 0.0, 0.0],
                dim4!(2, 3, 1, 1),
            ),
        ));
    }

    #[test]
    fn sum_forward_backward() {
        let x = mu::fill::<1, 1, 3, 2>(2.0);
        let z = x.sum();
        assert!(equal_arrays(z.data(), constant!(12.0; 1,1,1,1)));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(1.0; 3,2,1,1)));
    }

    #[test]
    fn add_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = x.add(&y);
        assert!(equal_arrays(
            z.data(),
            Array::new(&[5.0, 2.0, 2.0, 2.0, 5.0, 2.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(1.0; 3,2,1,1)));
        assert!(equal_arrays(y.grad().data(), constant!(1.0; 3,2,1,1)));
    }

    #[test]
    fn sub_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = x.sub(&y);
        assert!(equal_arrays(
            z.data(),
            Array::new(&[1.0, -2.0, -2.0, -2.0, 1.0, -2.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(1.0; 3,2,1,1)));
        assert!(equal_arrays(y.grad().data(), constant!(-1.0; 3,2,1,1)));
    }

    #[test]
    fn mul_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = x.mul(&y);
        assert!(equal_arrays(
            z.data(),
            Array::new(&[6.0, 0.0, 0.0, 0.0, 6.0, 0.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(2.0; 3,2,1,1)));
        assert!(equal_arrays(
            y.grad().data(),
            arrayfire::identity::<f32>(dim4!(3, 2, 1, 1)) * 3.0f32
        ));
    }

    #[test]
    fn div_forward_backward() {
        let x = mu::fill::<1, 1, 3, 2>(2.0);
        let y = mu::fill::<1, 1, 3, 2>(4.0);
        let z = x.div(&y);
        assert!(equal_arrays(z.data(), constant!(0.5; 3, 2, 1, 1)));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(0.25; 3,2,1,1)));
        assert!(equal_arrays(y.grad().data(), constant!(-0.125; 3,2,1,1)));
    }

    #[test]
    fn mm_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::eye::<1, 1, 2, 4>(2.0);
        let z = x.mm(&y);
        assert!(equal_arrays(
            z.data(),
            Array::new(
                &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dim4!(3, 4, 1, 1),
            ),
        ));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(2.0; 3,2,1,1)));
        assert!(equal_arrays(y.grad().data(), constant!(3.0; 2,4,1,1)));
    }

    #[test]
    fn pow_forward_backward() {
        let x = mu::fill::<1, 1, 3, 2>(2.0);
        let y = mu::fill::<1, 1, 3, 2>(3.0);
        let z = x.pow(&y);
        assert!(equal_arrays(z.data(), constant!(8.0; 3,2,1,1)));

        z.backward();
        assert!(equal_arrays(x.grad().data(), constant!(12.0; 3,2,1,1)));
        assert!(equal_arrays(y.grad().data(), constant!(5.5451775; 3,2,1,1)));
    }

    #[test]
    fn maximum_forward_backward() {
        let x = mu::eye::<1, 1, 3, 2>(3.0);
        let y = mu::fill::<1, 1, 3, 2>(2.0);
        let z = x.maximum(&y);
        assert!(equal_arrays(
            z.data(),
            Array::new(&[3.0, 2.0, 2.0, 2.0, 3.0, 2.0], dim4!(3, 2, 1, 1))
        ));

        z.backward();
        assert!(equal_arrays(
            x.grad().data(),
            Array::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dim4!(3, 2, 1, 1))
        ));
        assert!(equal_arrays(
            y.grad().data(),
            Array::new(&[0.0, 1.0, 1.0, 1.0, 0.0, 1.0], dim4!(3, 2, 1, 1))
        ));
    }
}

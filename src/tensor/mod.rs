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
use arrayfire::Array;

/// Defines operations on tensors, either `Constant` or `Variable`
pub trait Tensor {
    const BATCH: u64;
    const CHANNELS: u64;
    const HEIGHT: u64;
    const WIDTH: u64;

    /// Returns the batch size
    #[inline]
    fn batch(&self) -> u64 {
        Self::BATCH
    }

    /// Returns the number of channels
    #[inline]
    fn channels(&self) -> u64 {
        Self::CHANNELS
    }

    /// Returns the number of rows
    #[inline]
    fn height(&self) -> u64 {
        Self::HEIGHT
    }

    /// Returns the number of cols
    #[inline]
    fn width(&self) -> u64 {
        Self::WIDTH
    }

    /// Returns the tensor data as an `arrayfire` `Array`
    fn data(&self) -> Array<f32>;

    /// Does nothing
    #[must_use]
    #[inline]
    fn identity(&self) -> Self::Out
    where
        Self: SingleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }>,
    {
        let reverse = |df: &Array<f32>, _: &[Array<f32>]| df.clone();
        self.push_unary(self.data(), reverse, &[])
    }

    /// Changes the shape of the tensor to the given dimensions
    #[inline]
    fn reshape<const B: u64, const C: u64, const H: u64, const W: u64>(&self) -> Self::Out
    where
        Self: SingleParam<B, C, H, W>,
    {
        let reverse = |df: &Array<f32>, _: &[Array<f32>]| {
            arrayfire::moddims(
                df,
                arrayfire::dim4!(Self::HEIGHT, Self::WIDTH, Self::CHANNELS, Self::BATCH),
            )
        };
        self.push_unary(
            arrayfire::moddims(&self.data(), arrayfire::dim4!(H, W, C, B)),
            reverse,
            &[],
        )
    }

    /// Computes `sin(x)`
    #[must_use]
    #[inline]
    fn sin(&self) -> Self::Out
    where
        Self: SingleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }>,
    {
        let reverse = |df: &Array<f32>, args: &[Array<f32>]| df * arrayfire::cos(&args[0]);
        self.push_unary(arrayfire::sin(&self.data()), reverse, &[self.data()])
    }

    /// Computes `cos(x)`
    #[must_use]
    #[inline]
    fn cos(&self) -> Self::Out
    where
        Self: SingleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }>,
    {
        let reverse = |df: &Array<f32>, args: &[Array<f32>]| df * -arrayfire::sin(&args[0]);
        self.push_unary(arrayfire::cos(&self.data()), reverse, &[self.data()])
    }

    /// Perform the element-wise addition of two tensors
    #[inline]
    fn add<Y>(&self, other: &Y) -> Self::Out
    where
        Y: Tensor<
            BATCH = { Self::BATCH },
            CHANNELS = { Self::CHANNELS },
            HEIGHT = { Self::HEIGHT },
            WIDTH = { Self::WIDTH },
        >,
        Self:
            DoubleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }, Y>,
    {
        let reverse = |df: &Array<f32>, _: &[Array<f32>]| (df.clone(), df.clone());
        self.push_binary(
            other,
            arrayfire::add(&self.data(), &other.data(), false),
            reverse,
            &[],
        )
    }

    /// Perform the element-wise substraction of two tensors
    #[inline]
    fn sub<Y>(&self, other: &Y) -> Self::Out
    where
        Y: Tensor<
            BATCH = { Self::BATCH },
            CHANNELS = { Self::CHANNELS },
            HEIGHT = { Self::HEIGHT },
            WIDTH = { Self::WIDTH },
        >,
        Self:
            DoubleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }, Y>,
    {
        let reverse = |df: &Array<f32>, _: &[Array<f32>]| (df.clone(), -df.clone());
        self.push_binary(
            other,
            arrayfire::sub(&self.data(), &other.data(), false),
            reverse,
            &[],
        )
    }

    /// Perform the element-wise multiplication of two tensors
    #[inline]
    fn mul<Y>(&self, other: &Y) -> Self::Out
    where
        Y: Tensor<
            BATCH = { Self::BATCH },
            CHANNELS = { Self::CHANNELS },
            HEIGHT = { Self::HEIGHT },
            WIDTH = { Self::WIDTH },
        >,
        Self:
            DoubleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }, Y>,
    {
        let reverse = |df: &Array<f32>, args: &[Array<f32>]| (df * &args[1], df * &args[0]);
        self.push_binary(
            other,
            arrayfire::mul(&self.data(), &other.data(), false),
            reverse,
            &[self.data(), other.data()],
        )
    }

    /// Perform the element-wise division of two tensors
    #[inline]
    fn div<Y>(&self, other: &Y) -> Self::Out
    where
        Y: Tensor<
            BATCH = { Self::BATCH },
            CHANNELS = { Self::CHANNELS },
            HEIGHT = { Self::HEIGHT },
            WIDTH = { Self::WIDTH },
        >,
        Self:
            DoubleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Self::WIDTH }, Y>,
    {
        let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
            let (x, y) = (&args[0], &args[1]);
            (df / y, -(df * x / y / y))
        };
        self.push_binary(
            other,
            arrayfire::div(&self.data(), &other.data(), false),
            reverse,
            &[self.data(), other.data()],
        )
    }

    /// Perform the normal matrix multiplication of two tensors
    #[inline]
    fn mm<Y>(&self, other: &Y) -> Self::Out
    where
        Y: Tensor<BATCH = { Self::BATCH }, CHANNELS = { Self::CHANNELS }, HEIGHT = { Self::WIDTH }>,
        Self: DoubleParam<{ Self::BATCH }, { Self::CHANNELS }, { Self::HEIGHT }, { Y::WIDTH }, Y>,
    {
        // const CY: u64, Y: Tensor<B, L, C, CY>, Z: Tensor<B, L, R, CY>
        let reverse = |df: &Array<f32>, args: &[Array<f32>]| {
            (
                arrayfire::matmul(
                    df,
                    &args[1],
                    arrayfire::MatProp::NONE,
                    arrayfire::MatProp::TRANS,
                ),
                arrayfire::matmul(
                    &args[0],
                    df,
                    arrayfire::MatProp::TRANS,
                    arrayfire::MatProp::NONE,
                ),
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
            &[self.data(), other.data()],
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
        let z = x.reshape::<1, 1, 1, 6>();
        assert!(equal_arrays(
            z.data(),
            Array::new(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0], dim4!(1, 6, 1, 1))
        ));

        assert_eq!(z.batch(), 1);
        assert_eq!(z.channels(), 1);
        assert_eq!(z.height(), 1);
        assert_eq!(z.width(), 6);

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
}

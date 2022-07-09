use crate::tensor::{variable::Variable, Tensor};

/// Creates a variable tensor filled with the given value
#[must_use]
#[inline]
pub fn fill<const B: u64, const C: u64, const H: u64, const W: u64>(
    v: f32,
) -> Tensor<B, C, H, W, Variable> {
    Variable::from(arrayfire::constant!(v; H,W,C,B)).into()
}

/// Creates a variable tensor with the main diagonal filled with the given value, 0 everywhere else
#[must_use]
#[inline]
pub fn eye<const B: u64, const C: u64, const H: u64, const W: u64>(
    v: f32,
) -> Tensor<B, C, H, W, Variable> {
    Variable::from(v * arrayfire::identity::<f32>(arrayfire::dim4!(H, W, C, B))).into()
}

/// Creates a variable tensor with random values taken from a uniform distribution between [0,1]
#[must_use]
#[inline]
pub fn randu<const B: u64, const C: u64, const H: u64, const W: u64>(
) -> Tensor<B, C, H, W, Variable> {
    Variable::from(arrayfire::randu!(H, W, C, B)).into()
}

/// Creates a variable tensor with random values taken from a normal distribution centered at 0
#[must_use]
#[inline]
pub fn randn<const B: u64, const C: u64, const H: u64, const W: u64>(
) -> Tensor<B, C, H, W, Variable> {
    Variable::from(arrayfire::randn!(H, W, C, B)).into()
}

/// Creates a variable tensor from the given array of values
#[must_use]
#[inline]
pub fn custom<const B: u64, const C: u64, const H: u64, const W: u64>(
    values: &[f32],
) -> Tensor<B, C, H, W, Variable> {
    Variable::from(arrayfire::Array::new(values, arrayfire::dim4!(H, W, C, B))).into()
}

#[cfg(test)]
mod tests {
    use super::{custom, eye, fill, randn, randu};
    use crate::tensor::traits::Tensed;
    use crate::tests::equal_data;
    use arrayfire::{all_true_all, constant, dim4, identity, le};

    #[test]
    fn test_fill() {
        let x = fill::<1, 2, 3, 4>(2.0);
        assert!(equal_data(x.data(), constant!(2.0; 3,4,2,1)));
    }

    #[test]
    fn test_eye() {
        let x = eye::<1, 2, 3, 4>(2.0);
        assert!(equal_data(
            x.data(),
            identity::<f32>(dim4!(3, 4, 2, 1)) * 2.0f32
        ));
    }

    #[test]
    fn test_randu() {
        let x = randu::<1, 2, 3, 4>();
        assert!(all_true_all(&le(&x.data(), &constant!(1.0; 3,4,2,1), false)).0)
    }

    #[test]
    fn test_randn() {
        let x = randn::<1, 2, 3, 4>();
        assert!(all_true_all(&le(&x.data(), &constant!(5.0; 3,4,2,1), false)).0)
    }

    #[test]
    fn test_custom() {
        let x = custom::<1, 1, 1, 1>(&[1.0]);
        assert!(equal_data(x.data(), constant!(1.0;1,1,1,1)));
    }
}

//! # Mushin
//!
//! **Mushin** is a pure `Rust`, no-unsafe library for computing gradients on dynamic
//! computational graphs using
//! [reverse automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
//! In other words, what `PyTorch` is to `Python` is what `Mushin` is to `Rust`.
//!
//! All the operations on tensors use the excellent [arrayfire](https://arrayfire.com/)
//! library as a backend. Which means **Mushin can perform computations on any device**
//! (Nvidia CUDA GPUs, `OpenCL`, Intel MKL... ). Plus, all operations are checked at
//! compile time for mathematical correctness. I.e. You won't be able to add two tensors
//! of different shape/dimensions. The shape of the resulting tensors for all your
//! operations is tracked through the computation graph so in that regard we can offer
//! a guarantee that `Tensorflow` or `PyTorch` can't: **If it compiles, your computation
//! graph is guaranteed to be correct**
//!
//! ## Usage
//! ```rust
//! use mushin as mu;
//! use mu::Tensor;
//!
//! let x = mu::eye::<1, 1, 2, 3>(3.0).freeze();
//! let w = mu::randn::<1, 1, 3, 2>();
//! let b = mu::fill::<1, 1, 3, 3>(0.0);
//!
//! let z = w.mm(&x).add(&b);
//! z.backward();
//!
//! let dz_dw = w.grad();
//! let dz_db = b.grad();
//! ```
//! The code above is an example of a perceptron neural network layer, where we have an input (`x`)
//! that we treat as a constant and a set of variable (trainable) parameters, (`w`,`b`).
//! We then compute the output (`z`) as `WX + b`. All the operations are eagerly evaluated, so the
//! resulting tensor values are available at any time. Comparted to lazy evaluation, this has the
//! benefit that the built computation graph is trully dynamic, i.e. your graph operations can depend
//! on the result of previous operations.
//!
//! Mushin automatically keeps track of all the operations performed up until any given variable
//! and calling `backward()` in one of them traverses the computation graph in
//! [reverse mode](https://en.wikipedia.org/wiki/Automatic_differentiation) to accumulate the
//! gradients of all of its ancestor variables. By using the `grad()` method in any of them we can
//! now retrieve their gradients as new `Variable` tensor, which in turn can be used to compute
//! further gradients!
//!
//! It is quite possible the reader is more interested in the Deep Learning utilities of this
//! library rather than the raw auto-grad foundations.
//! By default, **Mushin** includes the [nn module](https://docs.rs/mushin/latest/mushin/nn/index.html)
//! that provides optimizers, activation functions, layers and losses ready to use to build neural network
//! modules. Checkout the module docs for instructions on how to use them.

#![deny(
    unsafe_code,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::module_name_repetitions,
    clippy::pattern_type_mismatch,
    clippy::shadow_unrelated,
    clippy::missing_inline_in_public_items
)]

#[cfg(feature = "nn")]
pub mod nn;

mod graph;
mod tensor;

use graph::{node::Node, tape::Tape};
use tensor::variable::Variable;

pub use tensor::Tensor;

/// Creates a `Variable` tensor filled with the given value
#[must_use]
#[inline]
pub fn fill<const B: u64, const L: u64, const R: u64, const C: u64>(
    v: f32,
) -> Variable<B, L, R, C> {
    let data = arrayfire::constant!(v; R,C,L,B);
    Variable::new(Tape::default(), Node::declaration(data))
}

/// Creates a `Variable` tensor with the main diagonal filled with the given value, 0 everywhere else
#[must_use]
#[inline]
pub fn eye<const B: u64, const L: u64, const R: u64, const C: u64>(v: f32) -> Variable<B, L, R, C> {
    let data = v * arrayfire::identity::<f32>(arrayfire::dim4!(R, C, L, B));
    Variable::new(Tape::default(), Node::declaration(data))
}

/// Creates a `Variable` tensor with random values taken from a uniform distribution between [0,1]
#[must_use]
#[inline]
pub fn randu<const B: u64, const L: u64, const R: u64, const C: u64>() -> Variable<B, L, R, C> {
    let data = arrayfire::randu!(R, C, L, B);
    Variable::new(Tape::default(), Node::declaration(data))
}

/// Creates a `Variable` tensor with random values taken from a normal distribution centered at 0
#[must_use]
#[inline]
pub fn randn<const B: u64, const L: u64, const R: u64, const C: u64>() -> Variable<B, L, R, C> {
    let data = arrayfire::randn!(R, C, L, B);
    Variable::new(Tape::default(), Node::declaration(data))
}

/// Creates a `Variable` tensor from the given array of values
#[must_use]
#[inline]
pub fn custom<const B: u64, const L: u64, const R: u64, const C: u64>(
    values: &[f32],
) -> Variable<B, L, R, C> {
    let data = arrayfire::Array::new(values, arrayfire::dim4!(R, C, L, B));
    Variable::new(Tape::default(), Node::declaration(data))
}

#[cfg(test)]
mod tests {
    use crate as mu;
    use arrayfire::{abs, all_true_all, constant, dim4, identity, le, Array};
    use mu::Tensor;

    pub(crate) fn equal_arrays(x: Array<f32>, y: Array<f32>) -> bool {
        all_true_all(&le(&abs(&(x - y)), &1e-15, false)).0
    }

    #[test]
    fn fill() {
        let x = mu::fill::<1, 2, 3, 4>(2.0);
        assert!(equal_arrays(x.data(), constant!(2.0; 3,4,2,1)));
    }

    #[test]
    fn eye() {
        let x = mu::eye::<1, 2, 3, 4>(2.0);
        assert!(equal_arrays(
            x.data(),
            identity::<f32>(dim4!(3, 4, 2, 1)) * 2.0f32
        ));
    }

    #[test]
    fn randu() {
        let x = mu::randu::<1, 2, 3, 4>();
        assert!(all_true_all(&le(&x.data(), &constant!(1.0; 3,4,2,1), false)).0)
    }

    #[test]
    fn randn() {
        let x = mu::randn::<1, 2, 3, 4>();
        assert!(all_true_all(&le(&x.data(), &constant!(3.0; 3,4,2,1), false)).0)
    }

    #[test]
    fn custom() {
        let x = mu::custom::<1, 1, 1, 1>(&[1.0]);
        assert!(equal_arrays(x.data(), constant!(1.0;1,1,1,1)));
    }
}

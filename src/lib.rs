//! # Mushin
//!
//! **Mushin** is a pure `Rust`, no-unsafe library for computing gradients on dynamic
//! computational graphs using
//! [reverse automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
//! In other words, what `PyTorch` is to `Python` is what `Mushin` is to `Rust`.
//!
//! This crate is backed by [arrayfire](https://arrayfire.com/) to perform the tensor
//! operations on any device, namely Nvidia CUDA GPUs, `OpenCL`, Intel MKL... On top of that,
//! all operations are checked at compile time for mathematical correctness.
//! I.e. You won't be able to add two tensors of different shape/dimensions.
//! The shapes of the resulting tensors for all your operations are tracked through the computation
//! graph at compilation time so in that regard we can offer a guarantee that `Tensorflow`
//! or `PyTorch` can't: **If it compiles, your computation graph is guaranteed to be correct**
//!
//! ## Usage
//! ```rust
//! #![feature(generic_const_exprs)]
//!
//! use mushin as mu;
//!
//! let x = mu::eye::<1, 1, 2, 3>(3.0).freeze();
//! let w = mu::randn::<1, 1, 3, 2>();
//! let b = mu::fill::<1, 1, 3, 3>(0.0);
//!
//! let z = mu::add(&mu::mm(&w, &x), &b);
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
//! and calling `backward()` in one of them traverses the computation graph in reverse mode
//! to accumulate the gradients of all of its ancestor variables.
//! By using the `grad()` method in any of them we can now retrieve their gradients as new variable
//! tensor, which in turn can be used to compute further gradients!
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
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(associated_const_equality)]

#[cfg(feature = "nn")]
pub mod nn;

mod gen;
mod graph;
mod ops;
mod tensor;

pub use gen::{custom, eye, fill, randn, randu};
pub use ops::{add, cos, div, mm, mul, reshape, sin, sub};

#[cfg(test)]
mod tests {
    use arrayfire::{abs, all_true_all, le, Array};

    pub(crate) fn equal_data(x: Array<f32>, y: Array<f32>) -> bool {
        all_true_all(&le(&abs(&(x - y)), &1e-6, false)).0
    }
}

//! # Mushin
//!
//! Mushin is a library for computing gradients on computational graphs using
//! reverse automatic differentiation. In other words, what Tensorflow is to
//! Python is what Mushin is to Rust.
//!
//! All the operations on tensors use the excellent [arrayfire](https://arrayfire.com/)
//! library as a backend. Which means **Mushin can perform computations on any device**
//! (Nvidia CUDA GPUs, `OpenCL`, Intel MKL... ). Plus, all operations are checked at
//! compile time for mathematical correctness. I.e. You won't be able to add two tensors
//! of different shape/dimensions. The shape of the resulting tensors for all your
//! operations is tracked through the computation graph so in that regard we can offer
//! a guarantee that Tensorflow can't: **If it compiles, your computation graph is
//! guaranteed to be correct**
//!
//! ## Usage
//!
//! All computational graphs start with a new context:
//! ```rust
//! # use mushin::Context;
//! let ctx = Context::new();
//! ```
//! The context contains the tape recording the computational graph as well as a storage
//! that lives through resets of the computational graph, to store for example tensors
//! whose values we want to keep, like trainable parameters.
//!
//! Once we have our context, we can start declaring tensors and use them in our operations:
//! ```rust
//! # use mushin::{Context, Values, add, matmul};
//! # let ctx = Context::new();
//! let x = ctx.constant::<1, 1, 2, 3>(Values::Eye(3.0));
//! let w = ctx.variable::<1, 1, 3, 2>(Values::Normal, Some("weights"));
//! let b = ctx.variable::<1, 1, 3, 3>(Values::Fill(0.0), Some("bias"));
//! let z = add(&b, &matmul(&w, &x));
//! ```
//! The code above is an example of a perceptron neural network layer, where we have an input (`x`)
//! that we treat as a constant and a set of persistent (trainable) parameters, (`w`,`b`).
//! We then compute the output (`z`) as `WX + b`. Being this a reverse automatic differentation
//! library, we are now of course interested on the gradients of the output with respect to the graph
//! variables, which are obtained as follows:
//! ```rust
//! # use mushin::{Context, Gradients, Values};
//! # let ctx = Context::new();
//! # let z = ctx.variable::<1, 1, 1, 1>(Values::Identity, None);
//! # let w = ctx.variable::<1, 1, 1, 1>(Values::Identity, None);
//! # let b = ctx.variable::<1, 1, 1, 1>(Values::Identity, None);
//! let grads = Gradients::compute(&z);
//! let dz_dw = grads.wrt(&w);
//! let dz_db = grads.wrt(&b);
//! ```

#![deny(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::module_name_repetitions,
    clippy::pattern_type_mismatch,
    clippy::shadow_unrelated,
    clippy::missing_inline_in_public_items
)]

mod context;
mod gradient;
mod ops;
mod tensor;

pub use context::Context;
pub use gradient::Gradients;
pub use ops::{add, div, matmul, mul, pow, sin, sub, sum};
pub use tensor::Values;

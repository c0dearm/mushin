//! This module exposes tooling for Deep Learning built upon the **Mushin** auto-grad core.
//!
//! ## Usage
//! ```rust
//! #![allow(incomplete_features)]
//! #![feature(associated_const_equality)]
//! #![feature(generic_const_exprs)]
//!
//! use mushin as mu;
//! use mu::nn::{layers::Linear, activations::relu, losses::mse, optimizers::SGD};
//!
//! let x = mu::eye::<16, 1, 1, 3>(1.0).freeze();
//! let y = mu::eye::<16, 1, 1, 5>(3.0).freeze();
//!
//! let linear = Linear::<3, 5, _>::randn();
//! let optim = SGD::new(&[linear.parameters()], 0.01);
//!
//! for _ in 0..5 {
//!     let z = relu(&linear.forward(&x));
//!     let loss = mse(&z, &y);
//!
//!     loss.backward();
//!     optim.step();
//!     loss.reset();
//! }
//! ```

pub mod activations;
pub mod layers;
pub mod losses;
pub mod optimizers;

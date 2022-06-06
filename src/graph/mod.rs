//! This module contains the data structures required to build the computation
//! graph that keeps track of all the operations computed up until certain
//! point, so that it can be traversed backwards to compute the gradients
//! of the `Variable` operands.
//!
//! The `Tape` struct is basically a set of `Node`s, and it is owned by `Variable` tensors.
//! It is the history of operations performed up until and including the existence
//! of its `Variable` owner. Everytime a new `Variable` is created, the `Tape` from the
//! parameters that originated it, is cloned and owned, the new `Node` associated to the
//! operation is then pushed into the `Tape`.
//!
//! A `Node` in the tape contains the data and the gradients of the `Variable` that
//! owns it, as well as a definition of the `Operation` that created that `Variable`.
//! Following the parameters of the operations backward is what allows to traverse
//! the the graph in reverse mode to perform the auto-differentiation.

pub mod node;
pub mod tape;

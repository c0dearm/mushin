pub mod function;
mod storage;
pub mod tape;

use crate::tensor::{Constant, Values, Variable};

use storage::Storage;
use tape::Tape;

/// Stores the computation graph and variable values that persist through different tape builds
pub struct Context {
    storage: Storage,
    tape: Tape,
}

impl Context {
    /// Creates a new `Context` with fresh storage and computation graph
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            tape: Tape::new(),
        }
    }

    // Start the computation graph from scratch, but keep the persistent values
    #[inline]
    pub fn reset(&self) {
        self.tape.reset();
    }

    /// Creates a new constant tensor
    #[must_use]
    #[inline]
    #[allow(clippy::unused_self)]
    pub fn constant<const B: u64, const L: u64, const R: u64, const C: u64>(
        &self,
        values: Values,
    ) -> Constant<B, L, R, C> {
        Constant::new(values)
    }

    /// Creates a new variable tensor, contributing to the computation graph
    #[inline]
    pub fn variable<'t, const B: u64, const L: u64, const R: u64, const C: u64>(
        &'t self,
        mut values: Values,
        name: Option<&'static str>,
    ) -> Variable<'t, B, L, R, C> {
        if let Some(key) = name {
            values = self.storage.get_or_create::<B, L, R, C>(key, values).into();
        }
        Variable::new(values, &self.tape)
    }
}

impl Default for Context {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

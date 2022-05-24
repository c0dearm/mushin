pub mod function;
mod storage;
mod tape;

use std::cell::Ref;

use arrayfire::{constant, dim4, identity, randn, randu};

use crate::tensor::{Class, Origin, Tensor, Values};

use function::Function;
use storage::Storage;
use tape::Tape;

/// Stores the computation graph (tape) of functions and persistent values thorugh different tape builds
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

    /// Creates a new tensor in the computation graph with the given parameters
    #[inline]
    pub fn tensor<const B: u64, const L: u64, const R: u64, const C: u64>(
        &self,
        values: Values,
        class: Class,
    ) -> Tensor<B, L, R, C> {
        let gen_values = |v| match v {
            Values::Identity => identity(dim4!(R, C, L, B)),
            Values::Uniform => randu!(R, C, L, B),
            Values::Normal => randn!(R, C, L, B),
            Values::Eye(x) => identity::<f32>(dim4!(R, C, L, B)) * x,
            Values::Fill(x) => constant!(x; R, C, L, B),
        };

        match class {
            Class::Constant => Tensor::new(gen_values(values), Origin::None, self),
            Class::Variable => Tensor::new(
                gen_values(values),
                Origin::Function(self.tape.push_function(Function::Nary)),
                self,
            ),
            Class::Persistent(key) => {
                let function = self.tape.push_function(Function::Nary);
                let value = self
                    .storage
                    .get_or_create(key, gen_values, values, function);
                Tensor::new(value, Origin::Function(function), self)
            }
        }
    }

    pub(crate) fn functions(&self) -> Ref<Vec<Function>> {
        self.tape.functions()
    }

    pub(crate) fn push_function(&self, function: Function) -> usize {
        self.tape.push_function(function)
    }

    pub(crate) fn tape_len(&self) -> usize {
        self.tape.len()
    }
}

impl Default for Context {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

use std::cell::RefCell;
use std::collections::HashMap;

use arrayfire::Array;

use crate::tensor::{Constant, Tensor, Values};

/// Stores the values of persistent variables across different computation graph builds.
pub struct Storage(RefCell<HashMap<&'static str, Array<f32>>>);

impl Storage {
    pub fn new() -> Self {
        Self(RefCell::new(HashMap::new()))
    }

    /// If the key-value already exists in the storage, return it. Otherwise create it
    pub fn get_or_create<const B: u64, const L: u64, const R: u64, const C: u64>(
        &self,
        key: &'static str,
        values: Values,
    ) -> Array<f32> {
        if self.0.borrow().contains_key(key) {
            self.0.borrow()[key].clone()
        } else {
            let value = Constant::<B, L, R, C>::gen_value(values);
            self.0.borrow_mut().insert(key, value.clone());
            value
        }
    }
}

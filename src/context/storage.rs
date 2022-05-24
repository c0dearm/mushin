use std::cell::RefCell;
use std::collections::HashMap;

use arrayfire::Array;

/// Stores the tensor value and a reference to the function that originated it.
/// Used by `Storage`
pub struct PersistentValue {
    pub(crate) value: Array<f32>,
    pub(crate) function: usize,
}

/// Stores the values of persistent tensors across different computation graph builds.
/// Used by the `Context`
pub struct Storage(RefCell<HashMap<&'static str, PersistentValue>>);

impl Storage {
    pub(crate) fn new() -> Self {
        Self(RefCell::new(HashMap::new()))
    }

    pub(crate) fn get_or_create<A, F: Fn(A) -> Array<f32>>(
        &self,
        key: &'static str,
        get_value_fn: F,
        arg: A,
        function: usize,
    ) -> Array<f32> {
        let mut map = self.0.borrow_mut();
        let item = map.get_mut(key);

        match item {
            Some(&mut PersistentValue {
                ref value,
                function: ref mut f,
            }) => {
                *f = function;
                value.clone()
            }
            None => {
                let value = get_value_fn(arg);
                map.insert(
                    key,
                    PersistentValue {
                        value: value.clone(),
                        function,
                    },
                );
                value
            }
        }
    }
}

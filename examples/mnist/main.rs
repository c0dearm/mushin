use mushin::{
    nn::{
        activations::relu,
        layers::{Conv2D, Dropout, Linear},
        ops::{flatten, MaxPool2D},
    },
    randn, Constant, Data, Node, Pair, Tensor, Variable,
};
use std::rc::Rc;

pub struct Model<T: Data = Variable> {
    conv1: Conv2D<1, 32, 3, 3, T>,
    conv2: Conv2D<32, 64, 3, 3, T>,
    dropout1: Dropout<T>,
    dropout2: Dropout<T>,
    fc1: Linear<9216, 128, T>,
    fc2: Linear<128, 10, T>,
}

impl Model<Variable> {
    pub fn new() -> Self {
        Model {
            conv1: Conv2D::randn(),
            conv2: Conv2D::randn(),
            dropout1: Dropout::prob(0.25),
            dropout2: Dropout::prob(0.5),
            fc1: Linear::randn(),
            fc2: Linear::randn(),
        }
    }

    // TODO: Derive
    pub fn freeze(self) -> Model<Constant> {
        Model {
            conv1: self.conv1.freeze(),
            conv2: self.conv2.freeze(),
            dropout1: self.dropout1.freeze(),
            dropout2: self.dropout2.freeze(),
            fc1: self.fc1.freeze(),
            fc2: self.fc2.freeze(),
        }
    }

    // TODO: Derive
    pub fn parameters(&self) -> [Rc<Node>; 4] {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.fc1.parameters(),
            self.fc2.parameters(),
        ]
    }
}

impl Model<Constant> {
    // TODO: Derive
    pub fn unfreeze(self) -> Model<Variable> {
        Model {
            conv1: self.conv1.unfreeze(),
            conv2: self.conv2.unfreeze(),
            dropout1: self.dropout1.unfreeze(),
            dropout2: self.dropout2.unfreeze(),
            fc1: self.fc1.unfreeze(),
            fc2: self.fc2.unfreeze(),
        }
    }
}

// TODO: Get rid of monster return type
impl<T: Data> Model<T> {
    fn forward<const B: u64, D: Pair<T>>(
        &self,
        x: &Tensor<B, 1, 28, 28, D>,
    ) -> Tensor<
        B,
        1,
        1,
        10,
        <<<<D as Pair<T>>::Output as Pair<T>>::Output as Pair<T>>::Output as Pair<T>>::Output,
    > {
        let x = relu(&self.conv1.forward(x));
        let x = relu(&self.conv2.forward(&x));
        let x = MaxPool2D::<2, 2, 2>::forward(&x);
        let x = self.dropout1.forward(&x);
        let x = flatten(&x);
        let x = self.fc1.forward(&x);
        let x = self.dropout2.forward(&x);
        self.fc2.forward(&x)
    }
}

fn main() {
    let model = Model::new();
    let x = randn::<16, 1, 28, 28>();
    let y = model.forward(&x);
}

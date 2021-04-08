use rand::distributions::Uniform;

use gamma::{activations::ReLu, layers::Dense, NeuralNetwork};
use gamma_derive::NeuralNetwork;

#[derive(NeuralNetwork, Debug)]
struct MyNetwork {
    input: Dense<ReLu, 2, 4>,
    hidden: Dense<ReLu, 4, 2>,
    output: Dense<ReLu, 2, 1>,
}

impl MyNetwork {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1.0..=1.0);

        MyNetwork {
            input: Dense::random(&mut rng, &dist),
            hidden: Dense::random(&mut rng, &dist),
            output: Dense::random(&mut rng, &dist),
        }
    }
}

fn main() {
    let nn = MyNetwork::new();
    println!("{:#?}", nn);

    let input = [0.0, 1.0];
    println!("Input: {:#?}", input);
    let output = nn.forward(input);
    println!("Output: {:#?}", output);
}

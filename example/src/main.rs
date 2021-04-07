use rand::distributions::Uniform;

use gamma::{activations::relu, layers::Dense, NeuralNetwork};
use gamma_derive::NeuralNetwork;

#[derive(NeuralNetwork, Debug)]
struct MyNetwork {
    input: Dense<2, 4>,
    hidden: Dense<4, 2>,
    output: Dense<2, 1>,
}

impl MyNetwork {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1.0..=1.0);

        MyNetwork {
            input: Dense::random(&mut rng, &dist, relu),
            hidden: Dense::random(&mut rng, &dist, relu),
            output: Dense::random(&mut rng, &dist, relu),
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

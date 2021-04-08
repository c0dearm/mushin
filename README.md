# MUSHIN

[![License](https://camo.githubusercontent.com/47069b7e06b64b608c692a8a7f40bc6915cf629c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d417061636865322e302532464d49542d626c75652e737667)](https://github.com/c0dearm/mushin/blob/master/COPYRIGHT)

Compile-time creation of neural networks with Rust

## Description

This is for now just a showcase project of what can be done with `const generics` introduced in [Rust 1.51](https://blog.rust-lang.org/2021/03/25/Rust-1.51.0.html). There is not a single usage of `vec` in this project (as of today).

MUSHIN allows the developer to build neural networks at compile-time, with preallocated arrays with well defined sizes. Aside from the performance improvement at runtime, another important benefit is that any possible mistake with the layout of the neural network, for example mismatching the inputs/outputs in the chain of layers, will be raised at compilation time.

This magic is accomplished thanks to two awesome Rust features:

1. `const generics`: The layer weights are defined as multi-dimensional arrays with generic sizes. Before this feature was introduced the only option was to use `vec` or go crazy and define different layer types for each possible number of weights!
2. `derive macro`: It is impossible to define an array or any other iterable of layers because it is an hetereogeneous set (different number of weights for each layer). To perform the forward pass you need to chain all the layers and propagate the input up to the lastest layer. The `NeuralNetwork` derive macro defines the `forward` method at compile-time, doing exactly that without any iteration.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
mushin = "0.1"
mushin_derive = "0.1"
```

And this is a very simple example to get you started:

```rust
use rand::distributions::Uniform;

use mushin::{activations::relu, layers::Dense, NeuralNetwork};
use mushin_derive::NeuralNetwork;

// Builds a neural network with 2 inputs and 1 output
// Made of 3 feed forward layers, you can have as many as you want and with any name
#[derive(NeuralNetwork, Debug)]
struct MyNetwork {
    input: Dense<2, 4>, // <# inputs, # outputs>
    hidden: Dense<4, 2>,
    output: Dense<2, 1>,
}

impl MyNetwork {
    // Initialize layer weights with a uniform distribution and set ReLU as activation function
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
    // Init the weights and perform a forward pass
    let nn = MyNetwork::new();
    println!("{:#?}", nn);

    let input = [0.0, 1.0];
    println!("Input: {:#?}", input);
    let output = nn.forward(input);
    println!("Output: {:#?}", output);
}
```

You may wonder how the `forward` method works. The `NeuralNetwork` derive macro defines it for you, and it looks like this for this particular example:

```rust
fn forward(&self, input: [f32; 2]) -> [f32; 1] {
    self.output.forward(self.hidden.forward(self.input.forward[input]))
}
```

Note how the forward method expects two input values because that's what the first (`input`) layer expects, and returns one single value because that's what the last layer (`output`) returns.

## Roadmap

- [x] Compile-time neural network consistency check
- [ ] Docs, CI/CD & Benchmarks
- [ ] Backward pass
- [ ] More layer types (convolution, dropout, lstm...)
- [ ] More activation functions (sigmoid, softmax...)
- [ ] Maaaybeee, CPU and/or GPU concurrency

## Contributing

If you find a vulnerability, bug or would like a new feature, [open a new issue](https://github.com/c0dearm/mushin/issues/new).

To introduce your changes into the codebase, submit a Pull Request.

Many thanks!

## License

MUSHIN is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.

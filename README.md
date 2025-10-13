# VeryDiff

## Polynomials

```julia
using VeryDiff, VNNLib
using VeryDiff: approximate_polynomial_iterative

model_file = "./test/examples/networks/mnist-net_256x4.onnx"
net = load_network(model_file)

lbs, ubs = zeros(784), ones(784)
z = Zonotope(lbs, ubs)

degree = 50
net_poly = approximate_polynomial_iterative(net, z, degree, verbosity=1)

```


## Code Structure

- `src`
    - `Network.jl`: 
        - Functionality for loading networks (also polynomial ones)
        - executing networks on concrete inputs
    - `Layers_Zonotope.jl`: executing individual networks with zonotopes (also polynomial ones)
    - `Layers_DiffZonotope.jl`: executing differential network with zonotopes
- `usage_examples.jl`: snippets of code for how to execute the algorithm

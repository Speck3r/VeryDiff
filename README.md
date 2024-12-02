# VeryDiff




## Code Structure

- `src`
    - `Network.jl`: 
        - Functionality for loading networks (also polynomial ones)
        - executing networks on concrete inputs
    - `Layers_Zonotope.jl`: executing individual networks with zonotopes (also polynomial ones)
    - `Layers_DiffZonotope.jl`: executing differential network with zonotopes
- `usage_examples.jl`: snippets of code for how to execute the algorithm

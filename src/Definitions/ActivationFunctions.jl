

"""
    `Φ(x)`

    Computes the cumulative distribution function of the standard normal distribution.
"""
Φ(x::Number) = 0.5 * (1 + erf(x /sqrt(2)))

"""
    `gelu(x)`

    Computes the Gaussian Error Linear Unit (GeLU) activation function.
"""
gelu = x -> x * Φ(x)

"""
    `dgelu(x)`

    Computes the derivative of the GeLU activation function.
"""
dgelu = x -> Φ(x) + x * (1/sqrt(2π) * exp(-0.5*x^2))

"""
    `d2gelu(x)`

    Computes the second derivative of the GeLU activation function.
"""
d2gelu = x -> (1/sqrt(2π) * exp(-0.5*x^2)) * (2 - x^2)

"""
Test if univariate polynomial represented by coefficients [p₀, p₁, ...]
represents a linear function.

args:
    v - vector of polynomial coefficients

returns:
    true ⟺ polynomial represents linear function
"""
islinear(v) = length(v) < 3 ? true : all(v[3:end] .== 0)
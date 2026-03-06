
"""
    `init_bounds_cache_approximation_domain(∂model::GeminiNetwork, P::PropState)`

    Initialize the bounds cache in the PropState with the approximation domain of the polynomial layers.
    This ensures that there is no error explosion when the current zonotope bounds are looser than the bounds used for the polynomial approximation.

    args:
    - `∂model` - the GeminiNetwork that may contain polynomial layers
    - `P` - the PropState whose bounds cache should be initialized (will be modified in-place)
"""
function init_bounds_cache_approximation_domain!(∂model::GeminiNetwork, P::PropState)
    for (i, Ls) in enumerate(∂model.diff_layers)
        L1 = VeryDiff.get_layer1(Ls)
        if typeof(L1) <: VeryDiff.Definitions.ONNXPoly
            bounds_cache = VeryDiff.Definitions.BoundsCache()
            bounds_cache.lower₁ = L1.l
            bounds_cache.upper₁ = L1.u
            bounds_cache.lower₂ = fill(-Inf, size(L1.l))
            bounds_cache.upper₂ = fill(Inf, size(L1.u))
            bounds_cache.∂lower = fill(-Inf, size(L1.l))
            bounds_cache.∂upper = fill(Inf, size(L1.u))
            bounds_cache.initialized = true
            P.task_bounds.bounds_cache[i] = bounds_cache
        end
    end
end
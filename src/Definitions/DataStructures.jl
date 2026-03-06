struct TaskBounds
    bounds_cache :: Dict{Int, BoundsCache}
    function TaskBounds()
        return new(Dict{Int, BoundsCache}())
    end
end


struct VerificationTask
    middle :: Vector{Float64}
    distance :: Vector{Float64}
    distance_indices :: Vector{Int}
    distance1_secondary :: Union{Nothing, Vector{Float64}}
    middle1_secondary :: Union{Nothing, Vector{Float64}}
    distance2_secondary :: Union{Nothing, Vector{Float64}}
    middle2_secondary :: Union{Nothing, Vector{Float64}}
    verification_status
    distance_bound :: Float64
    work_share :: Float64
    task_bounds :: TaskBounds
    """
        `VerificationTask(middle, distance, distance_indices, distance1_secondary, middle1_secondary, distance2_secondary, middle2_secondary, verification_status, distance_bound, work_share)`

        Create a `VerificationTask` with the specified parameters.

        args:
        - `middle`: The center point of the input space for the verification task.
        - `distance`: The perturbation distances for the input dimensions.
        - `distance_indices`: The indices of the input dimensions that are subject to perturbation (excluding dimensions with zero perturbation).
        - `distance1_secondary`: Optional secondary perturbation distances for the first network (used for HaloZonotopes)
        - `middle1_secondary`: Optional secondary center point for the first network (used for HaloZonotopes)
        - `distance2_secondary`: Optional secondary perturbation distances for the second network (used for HaloZonotopes)
        - `middle2_secondary`: Optional secondary center point for the second network (used for HaloZonotopes)
        - `verification_status`: The current verification status (e.g., :unknown, :verified, :falsified)
        - `distance_bound`: The bound on the distance for verification (used for early termination)
        - `work_share`: The share of the overall verification work that this task represents (used for load balancing in parallel verification)
    """
    function VerificationTask(middle :: Vector{Float64},
                              distance :: Vector{Float64},
                              distance_indices :: Vector{Int},
                              distance1_secondary :: Union{Nothing, Vector{Float64}},
                              middle1_secondary :: Union{Nothing, Vector{Float64}},
                              distance2_secondary :: Union{Nothing, Vector{Float64}},
                              middle2_secondary :: Union{Nothing, Vector{Float64}},
                              verification_status,
                              distance_bound :: Float64,
                              work_share :: Float64)
        return new(middle,
                    distance,
                    distance_indices,
                    distance1_secondary,
                    middle1_secondary,
                    distance2_secondary,
                    middle2_secondary,
                    verification_status,
                    distance_bound,
                    work_share,
                    TaskBounds()
                )
    end
    function VerificationTask(middle :: Vector{Float64},
                              distance :: Vector{Float64},
                              distance_indices :: Vector{Int},
                              distance1_secondary :: Union{Nothing, Vector{Float64}},
                              middle1_secondary :: Union{Nothing, Vector{Float64}},
                              distance2_secondary :: Union{Nothing, Vector{Float64}},
                              middle2_secondary :: Union{Nothing, Vector{Float64}},
                              verification_status,
                              distance_bound :: Float64,
                              work_share :: Float64,
                              task_bounds :: TaskBounds)
        return new(middle,
                    distance,
                    distance_indices,
                    distance1_secondary,
                    middle1_secondary,
                    distance2_secondary,
                    middle2_secondary,
                    verification_status,
                    distance_bound,
                    work_share,
                    task_bounds
                )
    end
end
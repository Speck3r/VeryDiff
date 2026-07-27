function σ_diff(x, y)
    return σ(x) .- σ(x .- y)
end

function extremest_σ_diff(y)
    return 2 .* σ(y ./ 2) .- 1
end

function extremest_σ_diff(x, y)
    return 2 .* σ(y ./ 2) .- 1
end

function σ_nondiff(x,y)
    return σ(x) .- σ(y)
end

function ∂σ_nondiff_∂y(y)
    return (.-exp.(.-y)) ./ ((1 .+ exp.(.-y)) .^2)
end

function ∂σ_nondiff_∂x(x)
    return (exp.(.-x)) ./ ((1 .+ exp.(.-x)) .^2)
end

function solve_∂σ_nondiff_∂y_upper(λ)
    return log.((-2 .* λ .+ sqrt.(4 .* λ .+ 1) .- 1) ./ (2 .* λ))
end

function solve_∂σ_nondiff_∂y_lower(λ)
    return log.((-2 .* λ .- sqrt.(4 .* λ .+ 1) .- 1) ./ (2 .* λ))
end

function solve_∂σ_nondiff_∂x_upper(λ)
    return log.((-2 .* λ .+ sqrt.(1 .- 4 .* λ) .+ 1) ./ (2 .* λ))
end

function solve_∂σ_nondiff_∂x_lower(λ)
    return log.((-2 .* λ .- sqrt.(1 .- 4 .* λ) .+ 1) ./ (2 .* λ))
end

function fslope_x_upper(tangent_points, lower₁, upper₁, lower₂, upper₂)
   return (σ_nondiff(tangent_points, lower₂) .- σ_nondiff(lower₁, lower₂)) ./ (tangent_points .- lower₁)
end

function fslope_x_lower(tangent_points, lower₁, upper₁, lower₂, upper₂)
    return (σ_nondiff(tangent_points, upper₂) .- σ_nondiff(upper₁, upper₂)) ./ (tangent_points .- upper₁)
end

function fslope_y_upper(tangent_points, lower₁, upper₁, lower₂, upper₂)
    return (σ_nondiff(upper₁, tangent_points) .- σ_nondiff(upper₁, upper₂)) ./ (tangent_points .- upper₂)
end

function fslope_y_lower(tangent_points, lower₁, upper₁, lower₂, upper₂)
    return (σ_nondiff(lower₁, tangent_points) .- σ_nondiff(lower₁, lower₂)) ./ (tangent_points .- lower₂)
end

function iterate_nondiff_x_upper!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        λ[iteration] = clamp.(fslope_x_upper((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),0,0.25)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂x_upper((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = upper₁[no_iteration]
    end
    λ[iteration] .= ∂σ_nondiff_∂x(@view tangent_points[iteration])
end

function iterate_nondiff_x_lower!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        λ[iteration] = clamp.(fslope_x_lower((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),0,0.25)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂x_lower((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = lower₁[no_iteration]
    end
    λ[iteration] .= ∂σ_nondiff_∂x(@view tangent_points[iteration])
end

function iterate_nondiff_y_upper!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        λ[iteration] = clamp.(fslope_y_upper((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),-0.25,0)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂y_upper((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = lower₂[no_iteration]
    end
    λ[iteration] .= ∂σ_nondiff_∂y(@view tangent_points[iteration])
end

function iterate_nondiff_y_lower!(tangent_points, λ, lower₁, upper₁, lower₂, upper₂)
    iteration = fill(true, length(lower₁))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        λ[iteration] = clamp.(fslope_y_lower((@view tangent_points[iteration]), (@view lower₁[iteration]), (@view upper₁[iteration]), (@view lower₂[iteration]), (@view upper₂[iteration])),-0.25,0)
        no_iteration = (abs.(λ) .< CUTOFF_SIGMOID_SLOPE)
        iteration = .!no_iteration
        tangent_points[iteration] = solve_∂σ_nondiff_∂y_lower((@view λ[iteration]))
        λ[no_iteration] .= 0
        tangent_points[no_iteration] = upper₂[no_iteration]
    end
    λ[iteration] .= ∂σ_nondiff_∂y(@view tangent_points[iteration])
end

function ∂σ_diff_∂y(x,y)
    return exp.(y .- x) ./ ((exp.(y .- x) .+ 1) .^2)
end

function ∂σ_diff_∂y_extremest(y)
    return exp.(y ./ 2) ./ ((exp.(y ./ 2) .+ 1) .^2)
end

function solve_extremest_∂σ_diff_∂y_upper(λ)
    return 2 .* log.((1 .- 2λ .+ sqrt.(1 .- 4λ)) ./ (2λ))
end

function solve_extremest_∂σ_diff_∂y_upper(x, λ)
    return 2 .* log.((1 .- 2λ .+ sqrt.(1 .- 4λ)) ./ (2λ))
end

function solve_extremest_∂σ_diff_∂y_lower(λ)
    return 2 .* log.((1 .- 2λ .- sqrt.(1 .- 4λ)) ./ (2λ))
end

function solve_extremest_∂σ_diff_∂y_lower(x, λ)
    return 2 .* log.((1 .- 2λ .- sqrt.(1 .- 4λ)) ./ (2λ))
end

function solve_∂σ_diff_∂y_upper(x, λ)
    return log.((exp.(x) .- 2λ .* exp.(x) .+ sqrt.(exp.(2x) .- 4λ .* exp.(2x))) ./ (2λ))    
end

function solve_∂σ_diff_∂y_lower(x, λ)
    return log.((exp.(x) .- 2λ .* exp.(x) .- sqrt.(exp.(2x) .- 4λ .* exp.(2x))) ./ (2λ))
end

function calc_slope_upper!(λ, tangent_points, upper₁, ∂lower, use_extremest, z_diff)
    z_diff[use_extremest] = extremest_σ_diff((@view tangent_points[use_extremest]))
    z_diff[.!use_extremest] = σ_diff((@view upper₁[.!use_extremest]), (@view tangent_points[.!use_extremest]))
    λ .= z_diff ./ (tangent_points .- ∂lower)
end

function calc_tangent_point_upper!(tangent_points, λ, upper₁, use_extremest)
    tangent_points[use_extremest] = solve_extremest_∂σ_diff_∂y_upper((@view λ[use_extremest])) 
    tangent_points[.!use_extremest] = solve_∂σ_diff_∂y_upper((@view upper₁[.!use_extremest]), (@view λ[.!use_extremest]))
end

function recalc_tangent_point_upper!(tangent_points, λ, changed_mask)
    tangent_points[changed_mask] = solve_extremest_∂σ_diff_∂y_upper((@view λ[changed_mask]))
end

function iterate_tangent_point_upper!(tangent_points_upper, upper_slope, upper₁, ∂lower)
    use_extremest = (upper₁ .>= (tangent_points_upper ./ 2))
    z_diff = zeros(length(upper_slope))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        calc_slope_upper!(upper_slope, tangent_points_upper, upper₁, ∂lower, use_extremest, z_diff)
        calc_tangent_point_upper!(tangent_points_upper, upper_slope, upper₁, use_extremest)
        changed_mask = (.!use_extremest .&& (upper₁ .>= (tangent_points_upper ./ 2)))
        recalc_tangent_point_upper!(tangent_points_upper, upper_slope, changed_mask)
        use_extremest .= use_extremest .| changed_mask
    end
    upper_slope[use_extremest] = ∂σ_diff_∂y_extremest((@view tangent_points_upper[use_extremest]))
    upper_slope[.!use_extremest] = ∂σ_diff_∂y((@view upper₁[.!use_extremest]) ,(@view tangent_points_upper[.!use_extremest]))
end

function calc_slope_lower!(λ, tangent_points, lower₁, ∂upper, use_extremest, z_diff)
    z_diff[use_extremest] = extremest_σ_diff((@view tangent_points[use_extremest]))
    z_diff[.!use_extremest] = σ_diff((@view lower₁[.!use_extremest]), (@view tangent_points[.!use_extremest]))
    λ .= z_diff ./ (tangent_points .- ∂upper)
end

function calc_tangent_point_lower!(tangent_points, λ, lower₁, use_extremest)
    tangent_points[use_extremest] = solve_extremest_∂σ_diff_∂y_lower((@view λ[use_extremest])) 
    tangent_points[.!use_extremest] = solve_∂σ_diff_∂y_lower((@view lower₁[.!use_extremest]), (@view λ[.!use_extremest]))
end

function recalc_tangent_point_lower!(tangent_points, λ, changed_mask)
    tangent_points[changed_mask] = solve_extremest_∂σ_diff_∂y_lower((@view λ[changed_mask]))
end 

function iterate_tangent_point_lower!(tangent_points_lower, lower_slope, lower₁, ∂upper)
    use_extremest = (lower₁ .<= (tangent_points_lower ./ 2))
    z_diff = zeros(length(lower_slope))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        calc_slope_lower!(lower_slope, tangent_points_lower, lower₁, ∂upper, use_extremest, z_diff)
        calc_tangent_point_lower!(tangent_points_lower, lower_slope, lower₁, use_extremest)
        changed_mask = (.!use_extremest .&& (lower₁ .<= (tangent_points_lower ./ 2)))
        recalc_tangent_point_lower!(tangent_points_lower, lower_slope, changed_mask)
        use_extremest .= use_extremest .| changed_mask
    end
    lower_slope[use_extremest] = ∂σ_diff_∂y_extremest((@view tangent_points_lower[use_extremest]))
    lower_slope[.!use_extremest] = ∂σ_diff_∂y((@view lower₁[.!use_extremest]) ,(@view tangent_points_lower[.!use_extremest]))
end

function calc_tangent_point_lower_opposing!(tangent_points_lower, upper_slope, lower₁, ∂lower)
    use_extremest = (lower₁ .<= (∂lower ./ 2))
    calc_tangent_point_lower!(tangent_points_lower, upper_slope, lower₁, use_extremest)
    changed_mask = (.!use_extremest .&& (lower₁ .<= (tangent_points_lower ./ 2)))
    recalc_tangent_point_lower!(tangent_points_lower, upper_slope, changed_mask)
    check_out_of_bounds = tangent_points_lower .< ∂lower
    tangent_points_lower[check_out_of_bounds] = ∂lower[check_out_of_bounds]
end

function calc_tangent_point_upper_opposing!(tangent_points_upper, lower_slope, upper₁, ∂upper)
    use_extremest = (upper₁ .>= (∂upper ./ 2))
    calc_tangent_point_upper!(tangent_points_upper, lower_slope, upper₁, use_extremest)
    changed_mask = (.!use_extremest .&& (upper₁ .>= (tangent_points_upper ./ 2)))
    recalc_tangent_point_upper!(tangent_points_upper, lower_slope, changed_mask) 
    check_out_of_bounds = tangent_points_upper .> ∂upper
    tangent_points_upper[check_out_of_bounds] = ∂upper[check_out_of_bounds]
end

function fν_any_all_any(λ, t_upper, t_lower, upper₁, lower₁, upper_func, lower_func, mask)
    return 0.5 .* (.-(@view λ[mask]) .* (@view t_upper[mask]) .+ upper_func((@view upper₁[mask]), (@view t_upper[mask]))
                                     .- (@view λ[mask]) .* (@view t_lower[mask]) .+ lower_func((@view lower₁[mask]), (@view t_lower[mask])))
end

function fμ_any_all_any(λ, t_upper, t_lower, upper₁, lower₁, upper_func, lower_func, mask)
    return 0.5 .* (.-(@view λ[mask]) .* (@view t_upper[mask]) .+ upper_func((@view upper₁[mask]), (@view t_upper[mask]))
                                     .+ (@view λ[mask]) .* (@view t_lower[mask]) .- lower_func((@view lower₁[mask]), (@view t_lower[mask])))
end

function fuse_extremest(lower₁, to_check, upper₁)
    return (lower₁ .<= (to_check ./ 2)) .&& ((to_check ./ 2) .<= upper₁)
end

function fuse_lower_normal(lower₁, to_check)
    return (lower₁ .> (to_check ./ 2))
end

function fuse_upper_normal(upper₁, to_check)
    return (upper₁ .< (to_check ./ 2))
end

function fslope_pos_all_any(to_check, upper₁, ∂lower, max_z_func, x, mask)
    return (max_z_func((@view x[mask]), (@view to_check[mask])) .- σ_diff((@view upper₁[mask]), (@view ∂lower[mask]))) ./ ((@view to_check[mask]) .- (@view ∂lower[mask]))
end

function fslope_neg_all_any(to_check, lower₁, ∂upper, min_z_func, x, mask)
    return (min_z_func((@view x[mask]), (@view to_check[mask])) .- σ_diff((@view lower₁[mask]), (@view ∂upper[mask]))) ./ ((@view to_check[mask]) .- (@view ∂upper[mask]))
end

function ftangent_point_pos_all_any_upper(λ, solve_derivative, x, mask)
    return solve_derivative(x[mask], λ[mask])
end

function iteration_pos_all_any_upper!(tangent_points, λ, lower₁, upper₁, ∂lower, ∂upper)
    use_extremest = falses(length(lower₁))
    use_lower_normal = falses(length(lower₁))
    use_upper_normal = falses(length(lower₁))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        use_extremest = fuse_extremest(lower₁, tangent_points, upper₁)
        use_lower_normal = fuse_lower_normal(lower₁, tangent_points)
        use_upper_normal = fuse_upper_normal(upper₁, tangent_points)
        λ[use_extremest] = fslope_pos_all_any(tangent_points, upper₁, ∂lower, extremest_σ_diff, upper₁, use_extremest)
        λ[use_upper_normal] = fslope_pos_all_any(tangent_points, upper₁, ∂lower, σ_diff, upper₁, use_upper_normal)
        λ[use_lower_normal] = fslope_pos_all_any(tangent_points, upper₁, ∂lower, σ_diff, lower₁, use_lower_normal)
        λ_is_zero = (λ .== 0.0)
        tangent_points[λ_is_zero] = ∂upper[λ_is_zero]
        tangent_points[use_extremest] = ftangent_point_pos_all_any_upper(λ, solve_extremest_∂σ_diff_∂y_upper, upper₁, use_extremest)
        tangent_points[use_upper_normal] = ftangent_point_pos_all_any_upper(λ, solve_∂σ_diff_∂y_upper, upper₁, use_upper_normal)
        tangent_points[use_lower_normal] = ftangent_point_pos_all_any_upper(λ, solve_∂σ_diff_∂y_upper, lower₁, use_lower_normal)
    end
    λ[use_extremest] = ∂σ_diff_∂y_extremest(@view tangent_points[use_extremest])
    λ[use_upper_normal] = ∂σ_diff_∂y(upper₁[use_upper_normal], tangent_points[use_upper_normal])
    λ[use_lower_normal] = ∂σ_diff_∂y(lower₁[use_lower_normal], tangent_points[use_lower_normal])
end

function ftangent_point_neg_all_any_lower(λ, solve_derivative, x, mask)
    return solve_derivative(x[mask], λ[mask])
end

function iteration_neg_all_any_lower!(tangent_points, λ, lower₁, upper₁, ∂upper, ∂lower)
    use_extremest = falses(length(lower₁))
    use_lower_normal = falses(length(lower₁))
    use_upper_normal = falses(length(lower₁))
    for i in 1:LOOP_ITERATIONS_DIFF_SIGMOID
        use_extremest = fuse_extremest(lower₁, tangent_points, upper₁)
        use_lower_normal = fuse_lower_normal(lower₁, tangent_points)
        use_upper_normal = fuse_upper_normal(upper₁, tangent_points)
        λ[use_extremest] = fslope_neg_all_any(tangent_points, lower₁, ∂upper, extremest_σ_diff, upper₁, use_extremest)
        λ[use_upper_normal] = fslope_neg_all_any(tangent_points, lower₁, ∂upper, σ_diff, upper₁, use_upper_normal)
        λ[use_lower_normal] = fslope_neg_all_any(tangent_points, lower₁, ∂upper, σ_diff, lower₁, use_lower_normal)
        λ_is_zero = (λ .== 0.0)
        tangent_points[λ_is_zero] = ∂lower[λ_is_zero]
        tangent_points[use_extremest] = ftangent_point_neg_all_any_lower(λ, solve_extremest_∂σ_diff_∂y_lower, upper₁, use_extremest)
        tangent_points[use_upper_normal] = ftangent_point_neg_all_any_lower(λ, solve_∂σ_diff_∂y_lower, upper₁, use_upper_normal)
        tangent_points[use_lower_normal] = ftangent_point_neg_all_any_lower(λ, solve_∂σ_diff_∂y_lower, lower₁, use_lower_normal)
    end
    λ[use_extremest] = ∂σ_diff_∂y_extremest(@view tangent_points[use_extremest])
    λ[use_upper_normal] = ∂σ_diff_∂y(upper₁[use_upper_normal], tangent_points[use_upper_normal])
    λ[use_lower_normal] = ∂σ_diff_∂y(lower₁[use_lower_normal], tangent_points[use_lower_normal])    
end

function fmin_z_value(lower₁, upper₁, ∂upper)
    return min.(σ_diff(upper₁, ∂upper), σ_diff(lower₁, ∂upper))   
end

function calc_lower_offset!(lower_offset, λ, lower₁, upper₁, ∂lower, ∂upper)
    option_a = solve_∂σ_diff_∂y_lower(lower₁, λ)
    option_b = solve_∂σ_diff_∂y_lower(upper₁, λ)
    offset_a = .-λ .* option_a .+ σ_diff(lower₁, option_a)
    offset_b = .-λ .* option_b .+ σ_diff(upper₁, option_b)
    offset_upperbound = .-λ .* ∂upper .+ fmin_z_value(lower₁, upper₁, ∂upper)
    offset_lowerbound = .-λ .* ∂lower .+ fmin_z_value(lower₁, upper₁, ∂lower)
    a_valid = (option_a .<= ∂upper) .&& (option_a .>= ∂lower)
    b_valid = (option_b .<= ∂upper) .&& (option_b .>= ∂lower)
    a_mask = a_valid .&& .!b_valid
    b_mask = .!a_valid .&& b_valid
    a_and_b = a_valid .&& b_valid
    neither = .!a_valid .&& .!b_valid
    lower_offset[a_mask] = min.(min.((@view offset_a[a_mask]), (@view offset_upperbound[a_mask])), @view offset_lowerbound[a_mask])
    lower_offset[b_mask] = min.(min.((@view offset_b[b_mask]), (@view offset_upperbound[b_mask])), @view offset_lowerbound[b_mask])
    lower_offset[a_and_b] = min.(min.((@view offset_a[a_and_b]), (@view offset_b[a_and_b])), min.((@view offset_upperbound[a_and_b]), @view offset_lowerbound[a_and_b]))
    lower_offset[neither] = min.(offset_upperbound[neither], offset_lowerbound[neither])
end

function fmax_z_value(lower₁, upper₁, ∂lower)
    return max.(σ_diff(upper₁, ∂lower), σ_diff(lower₁, ∂lower))   
end

function calc_upper_offset!(upper_offset, λ, lower₁, upper₁, ∂lower, ∂upper)
    option_a = solve_∂σ_diff_∂y_upper(lower₁, λ)
    option_b = solve_∂σ_diff_∂y_upper(upper₁, λ)
    offset_a = .-λ .* option_a .+ σ_diff(lower₁, option_a)
    offset_b = .-λ .* option_b .+ σ_diff(upper₁, option_b)
    offset_lowerbound = .-λ .* ∂lower .+ fmax_z_value(lower₁, upper₁, ∂lower)
    offset_upperbound = .-λ .* ∂upper .+ fmax_z_value(lower₁, upper₁, ∂upper)
    a_valid = (option_a .<= ∂upper) .&& (option_a .>= ∂lower)
    b_valid = (option_b .<= ∂upper) .&& (option_b .>= ∂lower)
    a_mask = a_valid .&& .!b_valid
    b_mask = .!a_valid .&& b_valid
    a_and_b = a_valid .&& b_valid
    neither = .!a_valid .&& .!b_valid
    upper_offset[a_mask] = max.(max.((@view offset_a[a_mask]), (@view offset_lowerbound[a_mask])), (@view offset_upperbound[a_mask]))
    upper_offset[b_mask] = max.(max.((@view offset_b[b_mask]), (@view offset_lowerbound[b_mask])), (@view offset_upperbound[b_mask]))
    upper_offset[a_and_b] = max.(max.((@view offset_a[a_and_b]), (@view offset_b[a_and_b])), max.((@view offset_lowerbound[a_and_b]),(@view offset_upperbound[a_and_b])))
    upper_offset[neither] = max.(offset_lowerbound[neither], offset_upperbound[neither])
end

function calc_max_z_value!(max_z_value, tangent_points, upper₁, lower₁)    
    use_extremest = fuse_extremest(lower₁, tangent_points, upper₁)
    use_upper_normal = fuse_upper_normal(upper₁, tangent_points)
    use_lower_normal = fuse_lower_normal(lower₁, tangent_points)
    max_z_value[use_extremest] = extremest_σ_diff((@view tangent_points[use_extremest]))
    max_z_value[use_upper_normal] = σ_diff((@view upper₁[use_upper_normal]), (@view tangent_points[use_upper_normal]))
    max_z_value[use_lower_normal] = σ_diff((@view lower₁[use_lower_normal]), (@view tangent_points[use_lower_normal]))
end

function calc_min_z_value!(min_z_value, tangent_points, upper₁, lower₁)    
    use_extremest = fuse_extremest(lower₁, tangent_points, upper₁)
    use_upper_normal = fuse_upper_normal(upper₁, tangent_points)
    use_lower_normal = fuse_lower_normal(lower₁, tangent_points)
    min_z_value[use_extremest] = extremest_σ_diff((@view tangent_points[use_extremest]))
    min_z_value[use_upper_normal] = σ_diff((@view upper₁[use_upper_normal]), (@view tangent_points[use_upper_normal]))
    min_z_value[use_lower_normal] = σ_diff((@view lower₁[use_lower_normal]), (@view tangent_points[use_lower_normal]))
end

function sigmoid_iteration_diff_relax(lower₁, upper₁, lower₂, upper₂, ∂lower, ∂upper)
    
    (zero_diff, c_all_all, all_c_all, all_all_neg, all_all_pos, neg_all_any, pos_all_any, any_all_any
    ) = get_sigmoid_selectors(lower₁, upper₁, lower₂, upper₂, ∂lower, ∂upper)

    dim = length(lower₁)
    slopes = zeros(dim)
    generators = zeros(dim)
    center_offsets = zeros(dim)

    #any_all_any case
    dim_any_all_any = length((@view upper₁[any_all_any]))
    λ_any_all_any = zeros(dim_any_all_any)
    ν_any_all_any = zeros(dim_any_all_any)
    μ_any_all_any = zeros(dim_any_all_any)
    
    let
        upper₁_any_all_any = @view upper₁[any_all_any]
        lower₁_any_all_any = @view lower₁[any_all_any]
        ∂upper_any_all_any = @view ∂upper[any_all_any]
        ∂lower_any_all_any = @view ∂lower[any_all_any]
        
        #upper
        upper_slope = zeros(dim_any_all_any)
        tangent_points_upper = copy(∂upper_any_all_any)
        max_∂pos = zeros(dim_any_all_any)
        max_derivatives_upper = zeros(dim_any_all_any)

        max_∂pos_mask_extremest = (upper₁_any_all_any .>= (∂upper_any_all_any ./ 2))
        max_∂pos_mask_normal = (upper₁_any_all_any .< (∂upper_any_all_any ./ 2))
        max_∂pos[max_∂pos_mask_extremest] = extremest_σ_diff((@view ∂upper_any_all_any[max_∂pos_mask_extremest]))
        max_∂pos[max_∂pos_mask_normal] = σ_diff((@view upper₁_any_all_any[max_∂pos_mask_normal]), (@view ∂upper_any_all_any[max_∂pos_mask_normal]))
        max_derivatives_upper[max_∂pos_mask_extremest] = ∂σ_diff_∂y_extremest((@view ∂upper_any_all_any[max_∂pos_mask_extremest]))
        max_derivatives_upper[max_∂pos_mask_normal] = ∂σ_diff_∂y((@view upper₁_any_all_any[max_∂pos_mask_normal]), (@view ∂upper_any_all_any[max_∂pos_mask_normal]))
        upper_secant_slope = max_∂pos ./ (∂upper_any_all_any .- ∂lower_any_all_any)
        use_secant_upper = (max_derivatives_upper .>= upper_secant_slope)
        need_iteration_upper = (max_derivatives_upper .< upper_secant_slope)
        upper_slope[use_secant_upper] = upper_secant_slope[use_secant_upper]
        iterate_tangent_point_upper!((@view tangent_points_upper[need_iteration_upper]), (@view upper_slope[need_iteration_upper]), (@view upper₁_any_all_any[need_iteration_upper]), (@view ∂lower_any_all_any[need_iteration_upper]))
        
        #lower
        lower_slope = zeros(dim_any_all_any)
        tangent_points_lower = copy(∂lower_any_all_any)
        min_∂neg = zeros(dim_any_all_any)
        max_derivatives_lower = zeros(dim_any_all_any)

        min_∂neg_mask_extremest = (lower₁_any_all_any .<= (∂lower_any_all_any ./ 2))
        min_∂neg_mask_normal = (lower₁_any_all_any .> (∂lower_any_all_any ./ 2))
        min_∂neg[min_∂neg_mask_extremest] = extremest_σ_diff((@view ∂lower_any_all_any[min_∂neg_mask_extremest]))
        min_∂neg[min_∂neg_mask_normal] = σ_diff((@view lower₁_any_all_any[min_∂neg_mask_normal]), (@view ∂lower_any_all_any[min_∂neg_mask_normal]))
        max_derivatives_lower[min_∂neg_mask_extremest] = ∂σ_diff_∂y_extremest(@view ∂lower_any_all_any[min_∂neg_mask_extremest])
        max_derivatives_lower[min_∂neg_mask_normal] = ∂σ_diff_∂y((@view lower₁_any_all_any[min_∂neg_mask_normal]), (@view ∂lower_any_all_any[min_∂neg_mask_normal]))
        lower_secant_slope = min_∂neg ./ (∂lower_any_all_any .- ∂upper_any_all_any)
        use_secant_lower = (max_derivatives_lower .>= lower_secant_slope)
        need_iteration_lower = (max_derivatives_lower .< lower_secant_slope)
        lower_slope[use_secant_lower] = lower_secant_slope[use_secant_lower]
        iterate_tangent_point_lower!((@view tangent_points_lower[need_iteration_lower]), (@view lower_slope[need_iteration_lower]), (@view lower₁_any_all_any[need_iteration_lower]), (@view ∂upper_any_all_any[need_iteration_lower]))

        use_upper_slope = upper_slope .<= lower_slope 
        use_lower_slope = upper_slope .> lower_slope
        λ_any_all_any = ifelse.(use_upper_slope, upper_slope, lower_slope) #final slope
        calc_tangent_point_lower_opposing!((@view tangent_points_lower[use_upper_slope]), (@view λ_any_all_any[use_upper_slope]), (@view lower₁_any_all_any[use_upper_slope]), (@view ∂lower_any_all_any[use_upper_slope]))
        calc_tangent_point_upper_opposing!((@view tangent_points_upper[use_lower_slope]), (@view λ_any_all_any[use_lower_slope]), (@view upper₁_any_all_any[use_lower_slope]), (@view ∂upper_any_all_any[use_lower_slope]))
        
        use_extremest_lower = (lower₁_any_all_any .<= (tangent_points_lower ./ 2))
        use_extremest_upper = (upper₁_any_all_any .>= (tangent_points_upper ./ 2))
        both = use_extremest_upper .&& use_extremest_lower
        upp = use_extremest_upper .&& .!use_extremest_lower
        low = .!use_extremest_upper .&& use_extremest_lower
        none = .!use_extremest_upper .&& .!use_extremest_lower
        ν_any_all_any[both] = fν_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, extremest_σ_diff, extremest_σ_diff, both)
        ν_any_all_any[upp] = fν_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, extremest_σ_diff, σ_diff, upp)
        ν_any_all_any[low] = fν_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, σ_diff, extremest_σ_diff, low)
        ν_any_all_any[none] = fν_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, σ_diff, σ_diff, none)
        μ_any_all_any[both] = fμ_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, extremest_σ_diff, extremest_σ_diff, both)
        μ_any_all_any[upp] = fμ_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, extremest_σ_diff, σ_diff, upp)
        μ_any_all_any[low] = fμ_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, σ_diff, extremest_σ_diff, low)
        μ_any_all_any[none] = fμ_any_all_any(λ_any_all_any, tangent_points_upper, tangent_points_lower, upper₁_any_all_any, lower₁_any_all_any, σ_diff, σ_diff, none)
    end
    generators[any_all_any] .= μ_any_all_any
    slopes[any_all_any] .= λ_any_all_any
    center_offsets[any_all_any] .= ν_any_all_any

    # pos_all_any case
    dim_pos_all_any = length((@view upper₁[pos_all_any]))
    λ_pos_all_any = zeros(dim_pos_all_any)
    ν_pos_all_any = zeros(dim_pos_all_any)
    μ_pos_all_any = zeros(dim_pos_all_any)

    let
        upper₁_pos_all_any = @view upper₁[pos_all_any]
        lower₁_pos_all_any = @view lower₁[pos_all_any]
        ∂upper_pos_all_any = @view ∂upper[pos_all_any]
        ∂lower_pos_all_any = @view ∂lower[pos_all_any]
        tangent_points_upper = copy(∂upper_pos_all_any)
        initial_derivatives = zeros(dim_pos_all_any)
        secant_slope = zeros(dim_pos_all_any) 
        lower_offset = zeros(dim_pos_all_any)
        max_z_value = zeros(dim_pos_all_any)

        use_extremest = fuse_extremest(lower₁_pos_all_any, ∂upper_pos_all_any, upper₁_pos_all_any)
        use_upper_normal = fuse_upper_normal(upper₁_pos_all_any, ∂upper_pos_all_any)
        use_lower_normal = fuse_lower_normal(lower₁_pos_all_any, ∂upper_pos_all_any)
        initial_derivatives[use_extremest] = ∂σ_diff_∂y_extremest((@view ∂upper_pos_all_any[use_extremest]))
        initial_derivatives[use_upper_normal] = ∂σ_diff_∂y((@view upper₁_pos_all_any[use_upper_normal]), (@view ∂upper_pos_all_any[use_upper_normal]))
        initial_derivatives[use_lower_normal] = ∂σ_diff_∂y((@view lower₁_pos_all_any[use_lower_normal]), (@view ∂upper_pos_all_any[use_lower_normal]))
        secant_slope[use_extremest] = fslope_pos_all_any(∂upper_pos_all_any, upper₁_pos_all_any, ∂lower_pos_all_any, extremest_σ_diff, upper₁_pos_all_any, use_extremest)
        secant_slope[use_upper_normal] = fslope_pos_all_any(∂upper_pos_all_any, upper₁_pos_all_any, ∂lower_pos_all_any, σ_diff, upper₁_pos_all_any, use_upper_normal)
        secant_slope[use_lower_normal] = fslope_pos_all_any(∂upper_pos_all_any, upper₁_pos_all_any, ∂lower_pos_all_any, σ_diff, lower₁_pos_all_any, use_lower_normal)
        use_secant = initial_derivatives .>= secant_slope
        need_iteration = initial_derivatives .< secant_slope
        λ_pos_all_any[use_secant] = secant_slope[use_secant]
        iteration_pos_all_any_upper!((@view tangent_points_upper[need_iteration]), (@view λ_pos_all_any[need_iteration]), 
        (@view lower₁_pos_all_any[need_iteration]), (@view upper₁_pos_all_any[need_iteration]), (@view ∂lower_pos_all_any[need_iteration]), (@view ∂upper_pos_all_any[need_iteration]))
        calc_lower_offset!(lower_offset, λ_pos_all_any, lower₁_pos_all_any, upper₁_pos_all_any, ∂lower_pos_all_any, ∂upper_pos_all_any)
        calc_max_z_value!(max_z_value, tangent_points_upper, upper₁_pos_all_any, lower₁_pos_all_any)

        ν_pos_all_any .= 0.5 .* (.-λ_pos_all_any .* tangent_points_upper .+ max_z_value .+ lower_offset)
        μ_pos_all_any .= 0.5 .* (.-λ_pos_all_any .* tangent_points_upper .+ max_z_value .- lower_offset)
    end
    generators[pos_all_any] .= μ_pos_all_any 
    slopes[pos_all_any] .= λ_pos_all_any
    center_offsets[pos_all_any] .= ν_pos_all_any

    # neg_all_any case
    dim_neg_all_any = length((@view upper₁[neg_all_any]))
    λ_neg_all_any = zeros(dim_neg_all_any)
    ν_neg_all_any = zeros(dim_neg_all_any)
    μ_neg_all_any = zeros(dim_neg_all_any)

    let 
        upper₁_neg_all_any = @view upper₁[neg_all_any]
        lower₁_neg_all_any = @view lower₁[neg_all_any]
        ∂upper_neg_all_any = @view ∂upper[neg_all_any]
        ∂lower_neg_all_any = @view ∂lower[neg_all_any]
        tangent_points_lower = copy(∂lower_neg_all_any)
        initial_derivatives = zeros(dim_neg_all_any)
        secant_slope = zeros(dim_neg_all_any)
        upper_offset = zeros(dim_neg_all_any)
        min_z_value = zeros(dim_neg_all_any)
        use_extremest = fuse_extremest(lower₁_neg_all_any, ∂lower_neg_all_any, upper₁_neg_all_any)
        use_upper_normal = fuse_upper_normal(upper₁_neg_all_any, ∂lower_neg_all_any)
        use_lower_normal = fuse_lower_normal(lower₁_neg_all_any, ∂lower_neg_all_any)

        initial_derivatives[use_extremest] = ∂σ_diff_∂y_extremest((@view ∂lower_neg_all_any[use_extremest]))
        initial_derivatives[use_upper_normal] = ∂σ_diff_∂y((@view upper₁_neg_all_any[use_upper_normal]), (@view ∂lower_neg_all_any[use_upper_normal]))
        initial_derivatives[use_lower_normal] = ∂σ_diff_∂y((@view lower₁_neg_all_any[use_lower_normal]), (@view ∂lower_neg_all_any[use_lower_normal]))
        secant_slope[use_extremest] = fslope_neg_all_any(∂lower_neg_all_any, lower₁_neg_all_any, ∂upper_neg_all_any, extremest_σ_diff, upper₁_neg_all_any, use_extremest)
        secant_slope[use_upper_normal] = fslope_neg_all_any(∂lower_neg_all_any, lower₁_neg_all_any, ∂upper_neg_all_any, σ_diff, upper₁_neg_all_any, use_upper_normal)
        secant_slope[use_lower_normal] = fslope_neg_all_any(∂lower_neg_all_any, lower₁_neg_all_any, ∂upper_neg_all_any, σ_diff, lower₁_neg_all_any, use_lower_normal)

        use_secant = initial_derivatives .>= secant_slope
        need_iteration = initial_derivatives .< secant_slope
        λ_neg_all_any[use_secant] = secant_slope[use_secant]
        iteration_neg_all_any_lower!((@view tangent_points_lower[need_iteration]), (@view λ_neg_all_any[need_iteration]), 
        (@view lower₁_neg_all_any[need_iteration]), (@view upper₁_neg_all_any[need_iteration]), (@view ∂upper_neg_all_any[need_iteration]), (@view ∂lower_neg_all_any[need_iteration]))
        calc_upper_offset!(upper_offset, λ_neg_all_any, lower₁_neg_all_any, upper₁_neg_all_any, ∂lower_neg_all_any, ∂upper_neg_all_any)
        calc_min_z_value!(min_z_value, tangent_points_lower, upper₁_neg_all_any, lower₁_neg_all_any)
        
        ν_neg_all_any .= 0.5 .* (.-λ_neg_all_any .* tangent_points_lower .+ min_z_value .+ upper_offset)
        μ_neg_all_any .= 0.5 .* (λ_neg_all_any .* tangent_points_lower .- min_z_value .+ upper_offset)

    end

    generators[neg_all_any] .= μ_neg_all_any 
    slopes[neg_all_any] .= λ_neg_all_any
    center_offsets[neg_all_any] .= ν_neg_all_any

    # all_all_neg or all_all_pos case
    all_all_np = (all_all_neg .|| all_all_pos)
    dim_all_all_np = length((@view upper₁[all_all_np]))
    λ_x_all_all_np = zeros(dim_all_all_np)
    λ_y_all_all_np = zeros(dim_all_all_np)
    ν_all_all_np = zeros(dim_all_all_np)
    μ_all_all_np = zeros(dim_all_all_np)

    let 
        upper₁_all_all_np = @view upper₁[all_all_np]
        lower₁_all_all_np = @view lower₁[all_all_np]
        upper₂_all_all_np = @view upper₂[all_all_np]
        lower₂_all_all_np = @view lower₂[all_all_np]

        λ_x = zeros(dim_all_all_np)
        λ_y = zeros(dim_all_all_np)

        λ_x_upper = zeros(dim_all_all_np)
        λ_x_lower = zeros(dim_all_all_np)
        λ_y_upper = zeros(dim_all_all_np)
        λ_y_lower = zeros(dim_all_all_np)

        tangent_points_x_upper = copy(upper₁_all_all_np)
        tangent_points_x_lower = copy(lower₁_all_all_np)
        tangent_points_y_upper = copy(lower₂_all_all_np)
        tangent_points_y_lower = copy(upper₂_all_all_np)

        intial_derivatives_x_upper = ∂σ_nondiff_∂x(upper₁_all_all_np)
        intial_derivatives_x_lower = ∂σ_nondiff_∂x(lower₁_all_all_np)
        intial_derivatives_y_upper = ∂σ_nondiff_∂y(lower₂_all_all_np)
        intial_derivatives_y_lower = ∂σ_nondiff_∂y(upper₂_all_all_np)

        secant_slope_x_upper = fslope_x_upper(upper₁_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)
        secant_slope_x_lower = fslope_x_lower(lower₁_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)
        secant_slope_y_upper = fslope_y_upper(lower₂_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)
        secant_slope_y_lower = fslope_y_lower(upper₂_all_all_np, lower₁_all_all_np, upper₁_all_all_np, lower₂_all_all_np, upper₂_all_all_np)

        use_secant_x_upper = (intial_derivatives_x_upper .>= secant_slope_x_upper)
        use_secant_x_lower = (intial_derivatives_x_lower .>= secant_slope_x_lower)
        use_secant_y_upper = (intial_derivatives_y_upper .<= secant_slope_y_upper)
        use_secant_y_lower = (intial_derivatives_y_lower .<= secant_slope_y_lower)

        need_iteration_x_upper = .!use_secant_x_upper
        need_iteration_x_lower = .!use_secant_x_lower
        need_iteration_y_upper = .!use_secant_y_upper
        need_iteration_y_lower = .!use_secant_y_lower

        λ_x_upper[use_secant_x_upper] = secant_slope_x_upper[use_secant_x_upper]
        λ_x_lower[use_secant_x_lower] = secant_slope_x_lower[use_secant_x_lower]
        λ_y_upper[use_secant_y_upper] = secant_slope_y_upper[use_secant_y_upper]
        λ_y_lower[use_secant_y_lower] = secant_slope_y_lower[use_secant_y_lower]

        iterate_nondiff_x_upper!((@view tangent_points_x_upper[need_iteration_x_upper]), (@view λ_x_upper[need_iteration_x_upper]), (@view lower₁_all_all_np[need_iteration_x_upper]),
        (@view upper₁_all_all_np[need_iteration_x_upper]), (@view lower₂_all_all_np[need_iteration_x_upper]), (@view upper₂_all_all_np[need_iteration_x_upper]))
        iterate_nondiff_x_lower!((@view tangent_points_x_lower[need_iteration_x_lower]), (@view λ_x_lower[need_iteration_x_lower]), (@view lower₁_all_all_np[need_iteration_x_lower]),
        (@view upper₁_all_all_np[need_iteration_x_lower]), (@view lower₂_all_all_np[need_iteration_x_lower]), (@view upper₂_all_all_np[need_iteration_x_lower]))
        iterate_nondiff_y_upper!((@view tangent_points_y_upper[need_iteration_y_upper]), (@view λ_y_upper[need_iteration_y_upper]), (@view lower₁_all_all_np[need_iteration_y_upper]),
        (@view upper₁_all_all_np[need_iteration_y_upper]), (@view lower₂_all_all_np[need_iteration_y_upper]), (@view upper₂_all_all_np[need_iteration_y_upper]))
        iterate_nondiff_y_lower!((@view tangent_points_y_lower[need_iteration_y_lower]), (@view λ_y_lower[need_iteration_y_lower]), (@view lower₁_all_all_np[need_iteration_y_lower]), 
        (@view upper₁_all_all_np[need_iteration_y_lower]), (@view lower₂_all_all_np[need_iteration_y_lower]), (@view upper₂_all_all_np[need_iteration_y_lower]))

        use_upper_x_slope = (λ_x_upper .<= λ_x_lower)
        use_upper_y_slope = (λ_y_upper .>= λ_y_lower)
        use_lower_x_slope = .!use_upper_x_slope
        use_lower_y_slope = .!use_upper_y_slope

        λ_x[use_upper_x_slope] = λ_x_upper[use_upper_x_slope]
        λ_x[use_lower_x_slope] = λ_x_lower[use_lower_x_slope]
        λ_y[use_upper_y_slope] = λ_y_upper[use_upper_y_slope]
        λ_y[use_lower_y_slope] = λ_y_lower[use_lower_y_slope]

        λ_x_zero = (λ_x .== 0)
        λ_y_zero = (λ_y .== 0)

        tangent_points_x_upper[use_lower_x_slope .&& λ_x_zero] = upper₁_all_all_np[use_lower_x_slope .&& λ_x_zero]
        tangent_points_x_upper[use_lower_x_slope .&& .!λ_x_zero] = solve_∂σ_nondiff_∂x_upper((@view λ_x[use_lower_x_slope .&& .!λ_x_zero])) 
        tangent_points_x_lower[use_upper_x_slope .&& λ_x_zero] = lower₁_all_all_np[use_upper_x_slope .&& λ_x_zero]
        tangent_points_x_lower[use_upper_x_slope .&& .!λ_x_zero] = solve_∂σ_nondiff_∂x_lower((@view λ_x[use_upper_x_slope .&& .!λ_x_zero]))
        
        tangent_points_y_upper[use_lower_y_slope .&& λ_y_zero] = lower₂_all_all_np[use_lower_y_slope .&& λ_y_zero]
        tangent_points_y_upper[use_lower_y_slope .&& .!λ_y_zero] = solve_∂σ_nondiff_∂y_upper((@view λ_y[use_lower_y_slope .&& .!λ_y_zero]))
        tangent_points_y_lower[use_upper_y_slope .&& λ_y_zero] = upper₂_all_all_np[use_upper_y_slope .&& λ_y_zero]
        tangent_points_y_lower[use_upper_y_slope .&& .!λ_y_zero] = solve_∂σ_nondiff_∂y_lower((@view λ_y[use_upper_y_slope .&& .!λ_y_zero]))
        
        λ_x_all_all_np .= λ_x
        λ_y_all_all_np .= λ_y

        ν_all_all_np .= 0.5 .* (.-λ_x .* tangent_points_x_lower .-λ_y .* tangent_points_y_lower .+ σ_nondiff(tangent_points_x_lower, tangent_points_y_lower) 
        .- λ_x .* tangent_points_x_upper .- λ_y .* tangent_points_y_upper .+ σ_nondiff(tangent_points_x_upper, tangent_points_y_upper))
        μ_all_all_np .= 0.5 .* (λ_x .* tangent_points_x_lower .+ λ_y .* tangent_points_y_lower .- σ_nondiff(tangent_points_x_lower, tangent_points_y_lower) 
        .- λ_x .* tangent_points_x_upper .- λ_y .* tangent_points_y_upper .+ σ_nondiff(tangent_points_x_upper, tangent_points_y_upper))

    end

    generators[all_all_np] .= μ_all_all_np 
    center_offsets[all_all_np] .= ν_all_all_np
    slope_x_all_all_np = zeros(dim)
    slope_y_all_all_np = zeros(dim)
    slope_x_all_all_np[all_all_np] .= λ_x_all_all_np
    slope_y_all_all_np[all_all_np] .= λ_y_all_all_np
    slope_all_all_np = (slope_x_all_all_np, slope_y_all_all_np)
    
    return slopes, generators, center_offsets, slope_all_all_np
end
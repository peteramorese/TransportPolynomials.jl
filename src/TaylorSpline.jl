struct TaylorSplineSegment{F}
    Ω_bounding_box::Hyperrectangle{Float64} # Current bounding box for the region
    duration::Float64                       # Time duration of the segment
    volume_function::F                      # Upper bound of the volume function for the segement 
end

struct TaylorSpline{F}
    segments::Vector{TaylorSplineSegment{F}}   # Array of Taylor spline segments ordered by time
end

function (ts::TaylorSpline)(t::Float64) 
    # Find the segment that contains the time t
    segment_time_end = 0.0
    for segment in ts.segments
        #println(" time_sum: $time_sum, t: $t")
        segment_time_end += segment.duration
        if t <= segment_time_end
            #println("computing volume at $(t - time_sum)")
            return segment.volume_function(t - segment_time_end + segment.duration)
        end
    end
    return ts.segments[end].volume_function(t - segment_time_end + ts.segments[end].duration)
    #error("Time $t exceeds the total duration of the Taylor spline.")
end

function create_box_taylor_spline(model::SystemModel, t_var::MultivariatePolynomials.AbstractVariable, vol_poly_degree::Int, init_set::Hyperrectangle{Float64}, duration::Float64; n_segments::Int=10, lagrangian_degree_inc::Int=1, geometric_bound=true)
    # Construct the volume polynomial components
    coefficients = compute_coefficients(model, vol_poly_degree + 1) # Add one for the bound
    t_monoms = monomials(t_var, 0:vol_poly_degree)
    taylor_scales = [1/factorial(i) for i in 0:vol_poly_degree]
    t_terms = t_monoms .* taylor_scales

    coeff_antiderivs = []
    for coeff in coefficients[1:end-1]
        p_antideriv = coeff
        for x_var in model.x_vars
            p_antideriv = antidifferentiate(p_antideriv, x_var)
        end
        push!(coeff_antiderivs, p_antideriv)
    end

    function compute_coeff_integrals(set::Hyperrectangle{Float64}, coeff_antideriv::AbstractPolynomialLike)
        # Compute the integral of the coefficient polynomial over the bounding box
        function coeff_antideriv_eval(x::Vector{Float64})
            subst = Dict(model.x_vars[i] => x[i] for i in 1:length(model.x_vars))
            return convert(Float64, subs(coeff_antideriv, subst...))
        end
        return evaluate_integral(coeff_antideriv_eval, set)
    end

    # Compute the upper bounds on the next coefficient
    nxt_coeff_ub = coeff_sos_bound(coefficients[end], lagrangian_degree_inc=lagrangian_degree_inc, bound_type=Upper)

    segment_duration = duration / n_segments

    segments = []

    vf_type = nothing

    Ω_bounding_box = init_set
    for k in 1:n_segments
        println("Creating segment $k of $n_segments with duration $segment_duration")

        # Compute the integral of each coefficient over the current bounding box
        coeff_integrals = [volume(Ω_bounding_box)] # coefficient of the zeroth order term is just the volume of the bounding box
        append!(coeff_integrals, [compute_coeff_integrals(Ω_bounding_box, coeff_antideriv) for coeff_antideriv in coeff_antiderivs])

        # Construct the volume polynomial for the segment
        volume_function_est = coeff_integrals' * t_terms

        # Taylor error bound polynomial
        taylor_error_bound = nxt_coeff_ub / factorial(vol_poly_degree + 1) * t_var^(vol_poly_degree + 1)

        ## Construct the bound polynomial for the segment
        #if geometric_bound
        #    # Find the maximum value of the taylor series over the interval (simple 1D polynomial bounding problem)
        #    V_max_estimate = sos_bound(volume_function.p, t_var, Hyperrectangle([0.0], [segment_duration]), 5, upper_bound=true)
        #    function ubound(t::Float64)
        #        return volume_function_est(t) / (1 - taylor_error_bound(t))
        #    end
        #else
        #end

        volume_function = TemporalPoly(t_var, volume_function_est + taylor_error_bound)

        vf_type = typeof(volume_function)

        # Add the segment to the spline
        push!(segments, TaylorSplineSegment(Ω_bounding_box, segment_duration, volume_function))

        if k < n_segments
            # Propagate the bounding box
            Ω_bounding_box = propagate_set(model, init_set=Ω_bounding_box, duration=segment_duration)
        end
    end
    ts = TaylorSpline{vf_type}(segments)
    return ts
end

function create_continuous_taylor_spline(model::SystemModel, t_var::MultivariatePolynomials.AbstractVariable, vol_poly_degree::Int, init_set::Hyperrectangle{Float64}, duration::Float64; n_segments::Int=10, lagrangian_degree_inc::Int=1, geometric_bound=true)
    # Construct the volume polynomial components
    coefficients = compute_coefficients(model, vol_poly_degree + 1) # Add one for the bound
    t_monoms = monomials(t_var, 0:vol_poly_degree)
    taylor_scales = [1/factorial(i) for i in 0:vol_poly_degree]
    t_terms = t_monoms .* taylor_scales

    coeff_antiderivs = []
    for coeff in coefficients[1:end-1]
        p_antideriv = coeff
        for x_var in model.x_vars
            p_antideriv = antidifferentiate(p_antideriv, x_var)
        end
        push!(coeff_antiderivs, p_antideriv)
    end

    function compute_coeff_integrals(set::Hyperrectangle{Float64}, coeff_antideriv::AbstractPolynomialLike)
        # Compute the integral of the coefficient polynomial over the bounding box
        function coeff_antideriv_eval(x::Vector{Float64})
            subst = Dict(model.x_vars[i] => x[i] for i in 1:length(model.x_vars))
            return convert(Float64, subs(coeff_antideriv, subst...))
        end
        return evaluate_integral(coeff_antideriv_eval, set)
    end

    # Compute the upper bounds on each coefficient
    last_coeff_upper_bound = coeff_sos_bound(coefficients[end], lagrangian_degree_inc=lagrangian_degree_inc, bound_type=Upper)
    coefficient_lower_bounds = [coeff_sos_bound(coeff, lagrangian_degree_inc=lagrangian_degree_inc, bound_type=Lower) for coeff in coefficients[1:end-1]]

    segment_duration = duration / n_segments

    segments = []

    vf_type = nothing

    Ω_bounding_box = init_set
    for k in 1:n_segments
        println("Creating segment $k of $n_segments with duration $segment_duration")

        # Compute the integral of each coefficient over the current bounding box
        coeff_integrals = [volume(Ω_bounding_box)] # coefficient of the zeroth order term is just the volume of the bounding box
        append!(coeff_integrals, [compute_coeff_integrals(Ω_bounding_box, coeff_antideriv) for coeff_antideriv in coeff_antiderivs])

        # Compute the derivative information using the preivous segment to compare to
        if k > 1
            prev_segment = segments[end]
            
            # Find the derivative bounds using the previous segments volume function
            prev_segment_vf = prev_segment.volume_function
            prev_seg_evals = []
            bounding_function = prev_segment_vf
            for i in 0:vol_poly_degree
                push!(prev_seg_evals, bounding_function(prev_segment.duration))
                if i < vol_poly_degree
                    bounding_function = differentiate(bounding_function)
                end
            end

            # Adjust the coeff integrals to account for the area difference
            volume_difference = coeff_integrals[1] - prev_seg_evals[1] # Difference between volume of the bounding box and previous segments predicted volume
            adjusted_coeff_integrals = [coeff_integrals[1]] # The zeroth order term is just the volume of the box, which will hopefully be worse than the previous segments volume
            append!(adjusted_coeff_integrals, [int - lb * volume_difference for (int, lb) in zip(coeff_integrals[2:end], coefficient_lower_bounds)])

            println("Adjusted coeff integrals: ", adjusted_coeff_integrals)
            println("Previous segment evals: ", prev_seg_evals)
            coeff_integrals = min.(adjusted_coeff_integrals, prev_seg_evals)
        end


        # Construct the volume polynomial for the segment
        volume_function_est = coeff_integrals' * t_terms

        # Taylor error bound polynomial
        taylor_error_bound = last_coeff_upper_bound / factorial(vol_poly_degree + 1) * t_var^(vol_poly_degree + 1)

        ## Construct the bound polynomial for the segment
        #if geometric_bound
        #    # Find the maximum value of the taylor series over the interval (simple 1D polynomial bounding problem)
        #    V_max_estimate = sos_bound(volume_function.p, t_var, Hyperrectangle([0.0], [segment_duration]), 5, upper_bound=true)
        #    function ubound(t::Float64)
        #        return volume_function_est(t) / (1 - taylor_error_bound(t))
        #    end
        #else
        #end

        volume_function = TemporalPoly(t_var, volume_function_est + taylor_error_bound)

        vf_type = typeof(volume_function)

        # Add the segment to the spline
        push!(segments, TaylorSplineSegment(Ω_bounding_box, segment_duration, volume_function))

        if k < n_segments
            # Propagate the bounding box
            Ω_bounding_box = propagate_set(model, init_set=Ω_bounding_box, duration=segment_duration)
        end
    end
    ts = TaylorSpline{vf_type}(segments)
    return ts
end
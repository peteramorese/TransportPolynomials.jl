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
        segment_time_end += segment.duration
        if t <= segment_time_end
            return segment.volume_function(t - segment_time_end + segment.duration)
        end
    end
    return ts.segments[end].volume_function(t - segment_time_end + ts.segments[end].duration)
    #error("Time $t exceeds the total duration of the Taylor spline.")
end

function create_box_taylor_spline(flow_pipe::RA.ReachSolution, model::SystemModel, vol_poly_degree::Int, init_set::Hyperrectangle{Float64}, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
    end

    n_segments = length(flow_pipe)
    segments = []

    for (k, reach_set) in enumerate(flow_pipe)
        segment_duration = reach_set.tspan
        Ω_bounding_box = compute_hyperrectangle(flow_pipe, k)

        println("Creating segment $k of $n_segments with duration $segment_duration")

        integ_poly = create_integ_poly(vol_poly, Ω_bounding_box)

        if rebound_each_segment
            bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff, Ω_bounding_box) 
        end

        volume_function = integ_poly + bound_poly
        push!(segments, TaylorSplineSegment{TemporalPoly}(Ω_bounding_box, segment_duration, volume_function))
    end
    return TaylorSpline{TemporalPoly}(segments)
end

function create_tamed_taylor_spline(flow_pipe::RA.ReachSolution, model::SystemModel, vol_poly_degree::Int, init_set::Hyperrectangle{Float64}, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
    end

    n_segments = length(flow_pipe)
    segments = []

    vf_coeffs = [Inf for _ in 1:(vol_poly_degree + 2)]

    for (k, reach_set) in enumerate(flow_pipe)
        segment_duration = reach_set.tspan
        Ω_curr_bounding_box = compute_hyperrectangle(flow_pipe, k)

        println("Creating segment $k of $n_segments with duration $segment_duration")

        integ_poly = create_integ_poly(vol_poly, Ω_bounding_box)

        if rebound_each_segment
            bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff, Ω_bounding_box) 
        end

        volume_function = integ_poly + bound_poly
        tamed_volume_function_coeffs = min.(vf_coeffs, volume_function.coeffs)
        tamed_volume_function = TemporalPoly(volume_function.t_deg, tamed_volume_function_coeffs)

        push!(segments, TaylorSplineSegment{TemporalPoly}(Ω_bounding_box, segment_duration, tamed_volume_function))
    end
    return TaylorSpline{TemporalPoly}(segments)
end
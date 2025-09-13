struct TaylorSplineSegment{F}
    Ω_bounding_box::Hyperrectangle # Current bounding box for the region
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
        if t < segment_time_end
            return min(segment.volume_function(abs(t - segment_time_end + segment.duration)), 1.0)
        end
    end
    return min(ts.segments[end].volume_function(abs(t - segment_time_end + ts.segments[end].duration)), 1.0)
end

function create_box_taylor_spline(flow_pipe::RA.AbstractFlowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    transition_sets::Vector{TransitionSet}()
    for reach_set in flow_pipe
        push!(transition_sets, TransitionSet(compute_hyperrectangle(flow_pipe, k), RA.tend(reach_set) - RA.tstart(reach_set)))
    end
    create_box_taylor_spline(Flowpipe(transition_sets), model, vol_poly_degree, rebound_each_segment)
end

function create_tamed_taylor_spline(flow_pipe::RA.AbstractFlowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    transition_sets::Vector{TransitionSet}()
    for reach_set in flow_pipe
        push!(transition_sets, TransitionSet(compute_hyperrectangle(flow_pipe, k), RA.tend(reach_set) - RA.tstart(reach_set)))
    end
    create_tamed_taylor_spline(Flowpipe(transition_sets), model, vol_poly_degree, rebound_each_segment)
end

function create_box_taylor_spline(flow_pipe::Flowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
    end

    segments = []

    for k in 1:length(flow_pipe.transition_sets)
        trans_set = flow_pipe.transition_sets[k]
        start_set = flow_pipe.start_sets[k]

        segment_duration = trans_set.duration

        integ_poly = create_integ_poly(vol_poly, start_set)

        if rebound_each_segment
            bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff, trans_set.set) 
        end

        volume_function = integ_poly + bound_poly
        push!(segments, TaylorSplineSegment{TemporalPoly}(trans_set.set, segment_duration, volume_function))
    end
    return TaylorSpline{TemporalPoly}(segments)
end

function create_tamed_taylor_spline(flow_pipe::Flowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
        roc_infemums = [lower_bound(coeff) for coeff in vol_poly.spatio_coeffs]
    end

    segments = []

    prev_vf = TemporalPoly(vol_poly_degree + 1, [Inf for _ in 1:(vol_poly_degree + 2)])

    # Calculate the upper bounds of the nxt coeff for all transition sets and use that for the bound poly
    trans_set_upper_bounds = [upper_bound(nxt_coeff, trans_set.set) for trans_set in flow_pipe.transition_sets]
    bound_poly_coeffs = zeros(Float64, vol_poly_degree + 2) # Add one since the bound poly is degree + 1
    bound_poly_coeffs[end] = maximum(trans_set_upper_bounds) / factorial(vol_poly_degree + 1)
    bound_poly = TemporalPoly(vol_poly_degree + 1, bound_poly_coeffs)

    for k in 1:length(flow_pipe.transition_sets)
        trans_set = flow_pipe.transition_sets[k]
        R = flow_pipe.start_sets[k]

        segment_duration = trans_set.duration

        # Calculate the volume rates of change over R
        roc = [integrate(coeff, R) for coeff in vol_poly.spatio_coeffs]

        if rebound_each_segment
            roc_infemums = [lower_bound(coeff, R) for coeff in vol_poly.spatio_coeffs]
        end

        # (upper bound) prediction of Ω volume using the previous volume function and the end of its duration
        if k > 1
            pred_Ω_volume = prev_vf(segments[end].duration)
            pred_volume_diff = clamp(volume(R) - pred_Ω_volume, 0.0, 1.0)
        else
            pred_Ω_volume = Inf
            pred_volume_diff = 0.0
        end
        # Worst case difference between integral of coefficients over R vs Ω_pred
        worst_case_roc_diff = pred_volume_diff * roc_infemums

        adjusted_roc = roc .- worst_case_roc_diff

        predictors = [prev_vf]
        pred_roc = [pred_Ω_volume]
        for _ in 1:vol_poly_degree 
            push!(predictors, differentiate(predictors[end]))
            push!(pred_roc, predictors[end](segment_duration))
        end

        # Compute the current volume function coefficients at the end of time time span
        tamed_roc = min.(adjusted_roc, pred_roc)
        taylor_scales = [1/factorial(i) for i in 0:vol_poly_degree]
        tamed_vf_coeffs = tamed_roc .* taylor_scales
        tamed_volume_function = TemporalPoly(vol_poly_degree, tamed_vf_coeffs) + bound_poly
        prev_vf = tamed_volume_function

        tamed_volume_function_cpy = TemporalPoly(tamed_volume_function.deg, copy(tamed_volume_function.coeffs))
        push!(segments, TaylorSplineSegment{TemporalPoly}(trans_set.set, segment_duration, tamed_volume_function_cpy))
    end
    return TaylorSpline{TemporalPoly}(segments)
end
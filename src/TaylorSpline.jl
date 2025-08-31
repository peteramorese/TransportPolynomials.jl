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
        if t <= segment_time_end
            return segment.volume_function(t - segment_time_end + segment.duration)
        end
    end
    return ts.segments[end].volume_function(t - segment_time_end + ts.segments[end].duration)
    #error("Time $t exceeds the total duration of the Taylor spline.")
end

function create_box_taylor_spline(flow_pipe::RA.AbstractFlowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
    end

    n_segments = length(flow_pipe)
    segments = []

    for (k, reach_set) in enumerate(flow_pipe)
        segment_duration = RA.tend(reach_set) - RA.tstart(reach_set)
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

function create_tamed_taylor_spline(flow_pipe::RA.AbstractFlowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
        coefficient_bounds = [lower_bound(coeff) for coeff in vol_poly.spatio_coeffs]
    end

    n_segments = length(flow_pipe)
    segments = []

    prev_vf = TemporalPoly(vol_poly_degree + 1, [Inf for _ in 1:(vol_poly_degree + 2)])

    for (k, reach_set) in enumerate(flow_pipe)
        segment_duration = RA.tend(reach_set) - RA.tstart(reach_set)
        R = compute_hyperrectangle(flow_pipe, k)
        #println("k: ", k, " R: ", R)

        println("Creating segment $k of $n_segments with duration $segment_duration")

        integ_poly = create_integ_poly(vol_poly, R)

        if rebound_each_segment
            bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff, R) 
            coefficient_infemums = [lower_bound(coeff, R) for coeff in vol_poly.spatio_coeffs]
        end

        volume_function = integ_poly + bound_poly
        println("integ poly coeffs: ", integ_poly.coeffs)
        
        # (upper bound) prediction of Ω volume using the previous volume function
        pred_Ω_volume = prev_vf(segment_duration)
        println("pred Ω volume: ", pred_Ω_volume)
        # vol(R) - vol(Ω_pred)
        if k > 1
            pred_volume_diff = volume(R) - pred_Ω_volume 
        else
            pred_volume_diff = 0.0
        end
        # Worst case difference between integral of coefficients over R vs Ω_pred
        println("Coefficient infemums: ", coefficient_infemums)
        worst_case_integ_diff = pred_volume_diff * coefficient_infemums

        println("worst case integ diff: ", worst_case_integ_diff)
        adjusted_vf_coeffs = integ_poly.coeffs .- worst_case_integ_diff

        predictors = [prev_vf]
        pred_Ω_int_coeffs = [pred_Ω_volume]
        for _ in 1:vol_poly_degree 
            push!(predictors, differentiate(predictors[end]))
            push!(pred_Ω_int_coeffs, predictors[end](segment_duration))
        end

        println("pred Ω int coeffs: ",pred_Ω_int_coeffs)
        println("prev coeffs: ", prev_vf.coeffs)
        println("adj VF coeffs: ", adjusted_vf_coeffs)
        #println("pred coeffs: ", pred_coeffs)
        println()

        # Compute the current volume function coefficients at the end of time time span

        tamed_volume_function_coeffs = min.(adjusted_vf_coeffs, pred_Ω_int_coeffs)
        #tamed_volume_function = TemporalPoly(volume_function.deg, tamed_volume_function_coeffs)
        tamed_volume_function = TemporalPoly(vol_poly_degree, tamed_volume_function_coeffs) + bound_poly
        prev_vf = tamed_volume_function

        push!(segments, TaylorSplineSegment{TemporalPoly}(R, segment_duration, tamed_volume_function))
    end
    return TaylorSpline{TemporalPoly}(segments)
end
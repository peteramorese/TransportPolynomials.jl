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
            #println("  seg coeffs: ", segment.volume_function.coeffs)
            #println("input dur: ", t - segment_time_end + segment.duration, " output: ", segment.volume_function(t - segment_time_end + segment.duration))
            return segment.volume_function(abs(t - segment_time_end + segment.duration))
        end
    end
    return ts.segments[end].volume_function(abs(t - segment_time_end + ts.segments[end].duration))
    #error("Time $t exceeds the total duration of the Taylor spline.")
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

    n_segments = length(flow_pipe.transition_sets)
    segments = []

    for (k, ts) in enumerate(flow_pipe.transition_sets)
        segment_duration = ts.duration
        R = ts.set

        #println("Creating segment $k of $n_segments with duration $segment_duration")

        integ_poly = create_integ_poly(vol_poly, R)

        if rebound_each_segment
            bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff, R) 
        end

        volume_function = integ_poly + bound_poly
        push!(segments, TaylorSplineSegment{TemporalPoly}(R, segment_duration, volume_function))
    end
    return TaylorSpline{TemporalPoly}(segments)
end

function create_tamed_taylor_spline(flow_pipe::Flowpipe, model::SystemModel, vol_poly_degree::Int, rebound_each_segment::Bool=true)
    vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, vol_poly_degree)

    if !rebound_each_segment
        bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff) 
        roc_infemums = [lower_bound(coeff) for coeff in vol_poly.spatio_coeffs]
    end

    n_segments = length(flow_pipe.transition_sets)
    segments = []

    prev_vf = TemporalPoly(vol_poly_degree + 1, [Inf for _ in 1:(vol_poly_degree + 2)])

    for (k, ts) in enumerate(flow_pipe.transition_sets)
        segment_duration = ts.duration
        R = ts.set
        #println("k: ", k, " R: ", R)

        #println("Creating segment $k of $n_segments with duration $segment_duration")

        #integ_poly = create_integ_poly(vol_poly, R)

        # Calculate the volume rates of change over R
        roc = [integrate(coeff, R) for coeff in vol_poly.spatio_coeffs]

        if rebound_each_segment
            bound_poly = create_bound_poly(vol_poly_degree, nxt_coeff, R) 
            roc_infemums = [lower_bound(coeff, R) for coeff in vol_poly.spatio_coeffs]
        end

        #volume_function = integ_poly + bound_poly
        #println("integ poly coeffs: ", integ_poly.coeffs)
        #println("ROC: ", roc)
        
        # (upper bound) prediction of Ω volume using the previous volume function and the end of its duration
        if k > 1
            pred_Ω_volume = prev_vf(segments[end].duration)
            pred_volume_diff = volume(R) - pred_Ω_volume 
        else
            pred_Ω_volume = Inf
            pred_volume_diff = 0.0
        end
        #println("  prev vf coeffs: ", prev_vf.coeffs)
        #println("pred Ω volume: ", pred_Ω_volume, " predicted at dur=", segment_duration)
        # vol(R) - vol(Ω_pred)
        # Worst case difference between integral of coefficients over R vs Ω_pred
        #println("ROC infemums: ", roc_infemums)
        worst_case_roc_diff = pred_volume_diff * roc_infemums

        #println("worst case integ diff: ", worst_case_roc_diff)
        adjusted_roc = roc .- worst_case_roc_diff
        #adjusted_vf_coeffs = integ_poly.coeffs .- worst_case_integ_diff

        predictors = [prev_vf]
        pred_roc = [pred_Ω_volume]
        for _ in 1:vol_poly_degree 
            push!(predictors, differentiate(predictors[end]))
            push!(pred_roc, predictors[end](segment_duration))
        end

        #println("pred roc: ",pred_roc)
        #println("adj roc: ", adjusted_roc)
        #println("pred coeffs: ", pred_coeffs)

        # Compute the current volume function coefficients at the end of time time span

        tamed_roc = min.(adjusted_roc, pred_roc)
        #tamed_volume_function = TemporalPoly(volume_function.deg, tamed_volume_function_coeffs)
        taylor_scales = [1/factorial(i) for i in 0:vol_poly_degree]
        tamed_vf_coeffs = tamed_roc .* taylor_scales
        tamed_volume_function = TemporalPoly(vol_poly_degree, tamed_vf_coeffs) + bound_poly
        #println("SEGMENT ", k, " tamed vf coeffs: ", tamed_volume_function.coeffs)
        prev_vf = tamed_volume_function

        #println()

        tamed_volume_function_cpy = TemporalPoly(tamed_volume_function.deg, copy(tamed_volume_function.coeffs))
        push!(segments, TaylorSplineSegment{TemporalPoly}(R, segment_duration, tamed_volume_function_cpy))
    end
    return TaylorSpline{TemporalPoly}(segments)
end
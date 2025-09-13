function plot_2D_region!(plt::Plots.Plot, region::Hyperrectangle{Float64}; vars::Tuple{Int, Int}=(1,2), alpha=0.5, color=:red, label="")
    if LazySets.dim(region) != 2
        region_mins = low(trans_set)
        region_maxes = high(trans_set)
        region_2D = Hyperrectangle(low=[region_mins[vars[1]], region_mins[vars[2]]], high=[region_maxes[vars[1]], region_maxes[vars[2]]])
    else
        region_2D = region
    end

    # Get center and radius
    c = region_2D.center
    r = region_2D.radius

    # Coordinates of the rectangle corners (counter-clockwise)
    x = [c[1] - r[1], c[1] + r[1], c[1] + r[1], c[1] - r[1], c[1] - r[1]]
    y = [c[2] - r[2], c[2] - r[2], c[2] + r[2], c[2] + r[2], c[2] - r[2]]

    return plot!(plt, x, y, seriestype=:shape, fillalpha=alpha, c=color, label=label) 
end

function plot_taylor_spline!(plt::Plots.Plot, ts::TaylorSpline{F}, duration::Float64; n_pts::Int=100, kwargs...) where {F}
    t_pts = range(0.0, duration, length=n_pts)
    ts_pts = [ts(t) for t in t_pts]
    plot!(plt, t_pts, ts_pts; kwargs...)
end

function plot_flowpipe!(plt::Plots.Plot, fp::Flowpipe; vars::Tuple{Int, Int}=(1,2), kwargs...)
    xlims!(plt, 0.0, 1.0)
    ylims!(plt, 0.0, 1.0)
    for trans_set in fp.transition_sets
        trans_set_mins = low(trans_set.set)
        trans_set_maxes = high(trans_set.set)
        subset = Hyperrectangle(low=[trans_set_mins[vars[1]], trans_set_mins[vars[2]]], high=[trans_set_maxes[vars[1]], trans_set_maxes[vars[2]]])
        plot_2D_region!(plt, subset; kwargs...)
    end
    return plt
end
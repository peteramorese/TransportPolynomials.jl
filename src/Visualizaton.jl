function plot_2D_region!(plt::Plots.Plot, region::Hyperrectangle{Float64}; vars::Tuple{Int, Int}=(1,2), alpha=0.5, color=:red, label="")
    if LazySets.dim(region) != 2
        region_mins = low(region)
        region_maxes = high(region)
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
    ts_pts = [clamp(ts(t), 0.0, 1.0) for t in t_pts]
    plot!(plt, t_pts, ts_pts; kwargs...)
end

function plot_flowpipe!(plt::Plots.Plot, fp::Flowpipe; vars::Tuple{Int, Int}=(1,2), n::Int=1, kwargs...)
    xlims!(plt, 0.0, 1.0)
    ylims!(plt, 0.0, 1.0)
    for i in 1:n:length(fp.transition_sets)
        trans_set = fp.transition_sets[i]
        trans_set_mins = low(trans_set.set)
        trans_set_maxes = high(trans_set.set)
        subset = Hyperrectangle(low=[trans_set_mins[vars[1]], trans_set_mins[vars[2]]], high=[trans_set_maxes[vars[1]], trans_set_maxes[vars[2]]])
        plot_2D_region!(plt, subset; kwargs...)
    end
    return plt
end

"""
Plot vector field for a 2D SystemModel

# Arguments
- `system::SystemModel`: The 2D system model to visualize
- `x_range::Tuple{Float64, Float64}`: Range for x-axis (default: (-2.0, 2.0))
- `y_range::Tuple{Float64, Float64}`: Range for y-axis (default: (-2.0, 2.0))
- `n_points::Int`: Number of grid points in each direction (default: 20)
- `scale::Float64`: Scale factor for arrow lengths (default: 0.3)
- `kwargs...`: Additional plotting arguments

# Returns
- `Plots.Plot`: The plot object containing the vector field
"""
function plot_vector_field!(plt::Plots.Plot, system::SystemModel; 
                          x_range::Tuple{Float64, Float64}=(-2.0, 2.0),
                          y_range::Tuple{Float64, Float64}=(-2.0, 2.0),
                          n_points::Int=20,
                          scale::Float64=0.3,
                          kwargs...)
    @assert dimension(system) == 2 "SystemModel must be 2D for vector field plotting"
    
    meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

    x, y = meshgrid(range(x_range[1], x_range[2], length=n_points), range(y_range[1], y_range[2], length=n_points))
    u_vals = []
    v_vals = []
    for (x_val, y_val) in zip(x, y)
        u_val, v_val = system([x_val, y_val])
        push!(u_vals, u_val[1])
        push!(v_vals, v_val[1])
    end
    return quiver!(plt, x, y, quiver=(u_vals, v_vals), kwargs...)
end

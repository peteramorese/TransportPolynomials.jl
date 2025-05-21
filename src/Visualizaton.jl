function plot_data(X, Y, title="")
    @assert size(X, 2) == 2
    p1 = scatter(X[:, 1], X[:, 2], Y[:, 1], markersize=4, title="f1(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f1")
    p2 = scatter(X[:, 1], X[:, 2], Y[:, 2], markersize=4, title="f2(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f2")
    display(plot!(p1, p2, layout=(1, 2), size=(800, 400), title=title))
end

function plot_polynomial_surface(p, x, y, xlim, ylim; n_points=50, kwargs...)
    # Create grid points
    xvals = range(xlim[1], xlim[2], length=n_points)
    yvals = range(ylim[1], ylim[2], length=n_points)
    
    # Store z values in matrix
    z_values = [p(x => x_val, y => y_val) for y_val in yvals, x_val in xvals]
    
    # Create the surface plot
    return Plots.surface(xvals, yvals, z_values; 
                   xlabel="$(x)", ylabel="$(y)", zlabel="p($(x),$(y))",
                   title="Polynomial Surface: $p",
                   kwargs...)
end

function plot_2D_pdf(pdf::Function, xlim, ylim; n_points::Int=100, kwargs...)
    # Create grid points
    xvals = range(xlim[1], xlim[2], length=n_points)
    yvals = range(ylim[1], ylim[2], length=n_points)
    
    # Store z values in matrix
    z_values = [pdf([x_val, y_val]) for y_val in yvals, x_val in xvals]
    
    # Create the surface plot
    return Plots.surface(xvals, yvals, z_values; 
                   xlabel="x₁ ", ylabel="x₂", zlabel="p(x₁, x₂)", kwargs...)
end

function plot_2D_erf_space_pdf(model::SystemModel, t_eval::Float64; n_points::Int=100, n_timesteps::Int=100)
    function euler_pdf(x_eval::Vector{Float64})
        return euler_density(x_eval, t_eval, model, n_timesteps)
    end
    return plot_2D_pdf(euler_pdf, (0, 1), (0, 1), n_points=n_points, title="ERF-space Euler pdf")
end

function plot_2D_pdf(model::SystemModel, t_eval::Float64, xlim, ylim; n_points::Int=100, n_timesteps::Int=100)
    d = Normal()
    function euler_pdf(x_eval::Vector{Float64})
        u_eval = cdf.(d, x_eval)
        volume_change = prod(pdf.(d, x_eval))
        return volume_change * euler_density(u_eval, t_eval, model, n_timesteps)
    end
    return plot_2D_pdf(euler_pdf, xlim, ylim, n_points=n_points, title="State space Euler pdf")
end

function plot_2D_erf_space_pdf(vol_poly::SpatioTemporalPoly, t_eval::Float64; n_points::Int=100)
    function vol_poly_pdf(x_eval::Vector{Float64})
        #println("density: ", density(x_eval, t_eval, vol_poly), " at x_eval: ", x_eval, " t_eval: ", t_eval)
        return density(x_eval, t_eval, vol_poly)
    end
    return plot_2D_pdf(vol_poly_pdf, (0, 1), (0, 1), n_points=n_points, title="ERF-space Vol-Poly pdf")
end

function plot_2D_pdf(vol_poly::SpatioTemporalPoly, t_eval::Float64, xlim, ylim; n_points::Int=100)
    d = Normal()
    function vol_poly_pdf(x_eval::Vector{Float64})
        u_eval = cdf.(d, x_eval)
        volume_change = prod(pdf.(d, x_eval))
        #println("density: ", density(u_eval, t_eval, vol_poly), " at x_eval: ", x_eval, " t_eval: ", t_eval)
        #println("volume change: ", volume_change, " density: ", density(x_eval, t_eval, vol_poly))
        return volume_change * density(u_eval, t_eval, vol_poly)
    end
    return plot_2D_pdf(vol_poly_pdf, xlim, ylim, n_points=n_points, title="State space Vol-Poly pdf")
end

function plot_2D_erf_space_vf(model::SystemModel; n_points::Int=100, scale::Float64=0.1)
    # Create grid points
    xvals = range(0, 1, length=n_points)
    yvals = range(0, 1, length=n_points)
    X, Y = repeat(xvals, outer=length(yvals)), repeat(yvals, inner=length(xvals))
    #print("X: ", X)
    inputs = @. (X, Y)
    
    f1_values = [convert(Float64, subs(model.f[1], Tuple(model.x_vars) => Tuple([x, y]))) for (x, y) in zip(X, Y)]
    f2_values = [convert(Float64, subs(model.f[2], Tuple(model.x_vars) => Tuple([x, y]))) for (x, y) in zip(X, Y)]

    return Plots.quiver(X, Y, quiver=(scale * f1_values, scale * f2_values); 
                   xlabel="x_1", ylabel="x_2", 
                   title="System Vector Field")
end

function plot_2D_region(plt, region::Hyperrectangle{Float64}; alpha=0.5)
    # Get base rectangle center and radius
    c = region.center
    r = region.radius
    if length(c) != 2 || length(r) != 2
        error("Expected a 2D Hyperrectangle")
    end

    # Compute base rectangle corners
    x0, y0 = c[1] - r[1], c[2] - r[2]
    x1, y1 = c[1] + r[1], c[2] + r[2]

    # z-limits from the plot
    z0, z1 = Plots.zlims(plt)

    # Define 8 corner points of the 3D box
    xp = [x0, x1, x0, x1, x0, x1, x0, x1]
    yp = [y0, y0, y1, y1, y0, y0, y1, y1]
    zp = [z0, z0, z0, z0, z1, z1, z1, z1]

    # Triangular faces: (1-based indices)
    connections = [
        (1,2,3), (4,2,3),       # bottom
        (5,6,7), (8,6,7),       # top
        (1,2,6), (1,5,6),       # front
        (2,4,8), (2,6,8),       # right
        (4,3,7), (4,8,7),       # back
        (3,1,5), (3,7,5)        # left
    ]

    # Plot the box using mesh3d!
    mesh3d!(plt, xp, yp, zp;
        connections,
        opacity=alpha,
        linecolor=:black,
        color=:lightblue,
        legend=false,
        linewidth=0,
    )

    return plt
end

#function plot_2D_region(plt, region::Hyperrectangle{Float64}; color="red", alpha=0.5)
#    center = region.center
#    radius = region.radius
#    xlims = (center[1] - radius[1], center[1] + radius[1])
#    ylims = (center[2] - radius[2], center[2] + radius[2])
#    
#    # For a 2D region in a 3D plot, we need to get the current z-limits
#    current_zlims = try
#        Plots.zlims(plt)
#    catch
#        (0.0, 1.0)  # Default if zlims cannot be retrieved
#    end
#    zmin, zmax = current_zlims
#    
#    # If zlims are not set (often returns (0,0)), use a default small value
#    if zmin == zmax
#        zmin = 0.0
#        zmax = 1.0
#    end
#    
#    # Instead of trying to plot complex 3D objects, let's use shape primitives
#    # Plot the bottom face (at zmin)
#    x_points = [xlims[1], xlims[2], xlims[2], xlims[1], xlims[1]]
#    y_points = [ylims[1], ylims[1], ylims[2], ylims[2], ylims[1]]
#    z_points = fill(zmin, 5)
#    
#    # Add the bottom face
#    plot!(plt, x_points, y_points, z_points, 
#          seriestype=:surface, 
#          seriescolor=color, 
#          fillalpha=alpha,
#          linewidth=0)
#    
#    # Add the top face (at zmax)
#    z_points = fill(zmax, 5)
#    plot!(plt, x_points, y_points, z_points, 
#          seriestype=:surface, 
#          seriescolor=color, 
#          fillalpha=alpha,
#          linewidth=0)
#    
#    # Add side walls (front side)
#    x_points = [xlims[1], xlims[2], xlims[2], xlims[1]]
#    y_points = [ylims[1], ylims[1], ylims[1], ylims[1]]
#    z_points = [zmin, zmin, zmax, zmax]
#    plot!(plt, x_points, y_points, z_points, 
#          seriestype=:surface, 
#          seriescolor=color, 
#          fillalpha=alpha,
#          linewidth=0)
#    
#    # Add side walls (back side)
#    y_points = [ylims[2], ylims[2], ylims[2], ylims[2]]
#    plot!(plt, x_points, y_points, z_points, 
#          seriestype=:surface, 
#          seriescolor=color, 
#          fillalpha=alpha,
#          linewidth=0)
#    
#    # Add side walls (left side)
#    x_points = [xlims[1], xlims[1], xlims[1], xlims[1]]
#    y_points = [ylims[1], ylims[2], ylims[2], ylims[1]]
#    plot!(plt, x_points, y_points, z_points, 
#          seriestype=:surface, 
#          seriescolor=color, 
#          fillalpha=alpha,
#          linewidth=0)
#    
#    # Add side walls (right side)
#    x_points = [xlims[2], xlims[2], xlims[2], xlims[2]]
#    plot!(plt, x_points, y_points, z_points, 
#          seriestype=:surface, 
#          seriescolor=color, 
#          fillalpha=alpha,
#          linewidth=0)
#    
#    return plt
#end
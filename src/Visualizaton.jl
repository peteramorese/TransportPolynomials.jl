function plot_data(X, Y, title="")
    @assert size(X, 2) == 2
    p1 = scatter(X[:, 1], X[:, 2], Y[:, 1], markersize=4, title="f1(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f1")
    p2 = scatter(X[:, 1], X[:, 2], Y[:, 2], markersize=4, title="f2(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f2")
    display(plot!(p1, p2, layout=(1, 2), size=(800, 400), title=title))
end

function plot_polynomial_surface(p, x, y, xlim, ylim; n_points=50, kwargs...)
    # Create grid points
    xpts = range(xlim[1], xlim[2], length=n_points)
    ypts = range(ylim[1], ylim[2], length=n_points)
    
    # Store z values in matrix
    z_values = [p(x => x_val, y => y_val) for y_val in ypts, x_val in xpts]
    
    # Create the surface plot
    return Plots.surface(xpts, ypts, z_values; 
                   xlabel="$(x)", ylabel="$(y)", zlabel="p($(x),$(y))",
                   kwargs...)
end

function plot_2D_pdf(pdf::Function, xlim, ylim; n_points::Int=100, kwargs...)
    # Create grid points
    xpts = range(xlim[1], xlim[2], length=n_points)
    ypts = range(ylim[1], ylim[2], length=n_points)
    
    # Store z values in matrix
    z_values = [pdf([x_val, y_val]) for y_val in ypts, x_val in xpts]
    
    # Create the surface plot
    return Plots.surface(xpts, ypts, z_values; 
                   xlabel="x₁ ", ylabel="x₂", zlabel="p(x₁, x₂)", kwargs...)
end

function plot_2D_erf_space_pdf(model::SystemModel, t_eval::Float64; n_points::Int=100, n_timesteps::Int=100)
    function euler_pdf(x_eval::Vector{Float64})
        return euler_density(x_eval, t_eval, model, n_timesteps=n_timesteps)
    end
    return plot_2D_pdf(euler_pdf, (0, 1), (0, 1), n_points=n_points, title="ERF-space Euler pdf")
end

function plot_2D_pdf(model::SystemModel, t_eval::Float64, xlim, ylim; n_points::Int=100, n_timesteps::Int=100)
    d = Normal()
    function euler_pdf(x_eval::Vector{Float64})
        u_eval = cdf.(d, x_eval)
        volume_change = prod(pdf.(d, x_eval))
        return volume_change * euler_density(u_eval, t_eval, model, n_timesteps=n_timesteps)
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
    xpts = range(0, 1, length=n_points)
    ypts = range(0, 1, length=n_points)
    X, Y = repeat(xpts, outer=length(ypts)), repeat(ypts, inner=length(xpts))
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

function plot_vol_poly_density_vs_time(x_eval::Vector{Float64}, vol_poly::SpatioTemporalPoly, duration::Float64; n_points::Int=100, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    for t in tpts
        append!(volpts, density(x_eval, t, vol_poly))
    end
    plt = plot(tpts, volpts, label="Vol Poly Density", xlabel="t", ylabel="p(x)", kwargs...)
    ylims!(plt, (0, min(10, maximum(volpts))))
    return plt
end

function plot_vol_poly_density_vs_time(x_eval::Vector{Float64}, vol_poly::SpatioTemporalPoly, bound_poly::TemporalPoly, duration::Float64; n_points::Int=100, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    errorpts = []
    for t in tpts
        append!(volpts, density(x_eval, t, vol_poly))
        append!(errorpts, bound_poly(t))
    end
    plt = plot(tpts, volpts, label="Vol Poly Density", xlabel="t", ylabel="p(x)", kwargs...)
    plot!(plt, tpts, volpts + errorpts, color=:black, linestyle=:dash)
    plot!(plt, tpts, volpts - errorpts, color=:black, linestyle=:dash)
    ylims!(plt, (0, min(10, maximum(volpts + errorpts))))
    return plt
end

function plot_euler_density_vs_time(x_eval::Vector{Float64}, model::SystemModel, duration::Float64; n_points::Int=100, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    for t in tpts
        append!(volpts, euler_density(x_eval, t, model))
    end
    return plot(tpts, volpts, label="Monte Carlo Density", xlabel="t", ylabel="p(x)", kwargs...)
end

function plot_euler_density_vs_time(plt::Plots.Plot, x_eval::Vector{Float64}, model::SystemModel, duration::Float64; n_points::Int=100, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    for t in tpts
        append!(volpts, euler_density(x_eval, t, model))
    end
    return plot(plt, tpts, volpts, label="Monte Carlo Density", xlabel="t", ylabel="p(x)", kwargs...)
end

function plot_integ_poly_prob_vs_time(region::Hyperrectangle{Float64}, integ_poly::SpatioTemporalPoly, duration::Float64; n_points::Int=100, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    for t in tpts
        append!(volpts, probability(region, t, integ_poly))
    end
    plt = plot(tpts, volpts, label="Integ Poly Probability", xlabel="t", ylabel="P(x∈R)", kwargs...)
    ylims!(plt, (0, 1))
    return plt
end

function plot_integ_poly_prob_vs_time(region::Hyperrectangle{Float64}, integ_poly::SpatioTemporalPoly, bound_poly::TemporalPoly, duration::Float64; plt::Union{Plots.Plot, Nothing}=nothing, n_points::Int=100, geometric::Bool=false, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    volpts_u_bound = []
    for t in tpts
        if geometric
            prob, u_bound = probability_and_geometric_ubound(region, t, integ_poly, bound_poly)
        else
            prob, u_bound = probability_and_ubound(region, t, integ_poly, bound_poly)
        end
        append!(volpts, prob)
        append!(volpts_u_bound, u_bound)
    end
    if isnothing(plt)
        plt = plot(tpts, volpts, label="Integ Poly Probability", xlabel="t", ylabel="P(x∈R)", kwargs...)
    else
        plot!(plt, tpts, volpts, label="Integ Poly Probability", xlabel="t", ylabel="P(x∈R)", kwargs...)
    end
    plot!(plt, tpts, volpts_u_bound, color=:black, linestyle=:dash, label="Upper bound")
    ylims!(plt, (0, 1))
    return plt
end

function plot_euler_mc_prob_vs_time(region::Hyperrectangle{Float64}, model::SystemModel, duration::Float64; n_points::Int=100, n_samples::Int=1000, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    for t in tpts
        append!(volpts, mc_euler_probability(region, t, model, n_samples=n_samples))
    end
    plt = plot(tpts, volpts, label="Monte Carlo Probability", xlabel="t", ylabel="P(x∈R)", kwargs...)
    ylims!(plt, (0, 1))
    return plt
end

function plot_euler_mc_prob_vs_time(plt::Plots.Plot, region::Hyperrectangle{Float64}, model::SystemModel, duration::Float64; n_points::Int=100, n_samples::Int=1000, kwargs...)
    tpts = range(0.0, duration, length=n_points)
    volpts = []
    for t in tpts
        append!(volpts, mc_euler_probability(region, t, model, n_samples=n_samples))
    end
    return plot(plt, tpts, volpts, label="Monte Carlo Density", xlabel="t", ylabel="P(x∈R)", kwargs...)
end
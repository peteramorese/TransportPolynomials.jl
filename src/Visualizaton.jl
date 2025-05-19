module Visualization

using Plots

include("DataStructures.jl")

#if !@isdefined(DataStructures)
#    include("DataStructures.jl")
#end
using .DataStructures

export plot_data, plot_polynomial_surface, plot_2D_pdf, plot_2D_erf_space_pdf, plot_2D_erf_space_vf

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
    return plot_2D_pdf(euler_pdf, (0, 1), (0, 1), n_points=n_points, title="ERF-space pdf")
end

function plot_2D_pdf(model::SystemModel, t_eval::Float64, xlim, ylim; n_points::Int=100, n_timesteps::Int=100)
    d = Normal()
    function euler_pdf(x_eval::Vector{Float64})
        u_eval = cdf.(d, x_eval)
        volume_change = prod(pdf.(d, x_eval))
        return volume_change * euler_density(u_eval, t_eval, model, n_timesteps)
    end
    return plot_2D_pdf(euler_pdf, xlim, ylim, n_points=n_points, title="State space PDF")
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

end # module
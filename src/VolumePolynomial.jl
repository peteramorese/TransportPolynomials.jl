using DynamicPolynomials
using MultivariatePolynomials
using LazySets
using IterTools
using Distributions
using LinearAlgebra

include("Visualizaton.jl")
include("DataStructures.jl")

using .DataStructures
using .Visualization

begin
function divergence(x::Vector{<:MultivariatePolynomials.AbstractVariable}, p::Vector{<:MultivariatePolynomials.AbstractPolynomialLike})
    divergence_poly = 0
    for i in 1:length(x)
        divergence_poly += differentiate(p[i], x[i])
    end
    return divergence_poly
end

function reynolds_operator(x::Vector{<:MultivariatePolynomials.AbstractVariable}, Φ::AbstractPolynomialLike, v::Vector{AbstractPolynomialLike})
    Φ_scaled_field = Φ .* v
    return divergence(x, Φ_scaled_field)
end

function compute_coefficients(model::SystemModel, degree::Int=1)
    Φ_i = 1
    coefficients = []
    for i in 1:degree
        Φ_ip1 = reynolds_operator(x, Φ_i, model)
        Φ_i = Φ_ip1
        push!(coefficients, Φ_i)
    end
    return coefficients
end

function create_vol_poly(
        model::SystemModel, 
        t::Variable,
        degree::Int=1)

    coefficients = compute_coefficients(model, degree)

    t_monoms = monomials(t, 1:degree)

    return t_monoms' * coefficients
end

function create_integrator_polynomial(vol_poly::SpatioTemporalPoly)
    # Create the antiderivative polynomial
    p_antideriv = vol_poly.p
    for x_var in vol_poly.x_vars
        p_antideriv = antidifferentiate(p_antideriv, x_var)
    end

    return SpatioTemporalPoly(vol_poly.x_vars, vol_poly.t_var, p_antideriv)
end

function evaluate_integral(antideriv, region::Hyperrectangle{Float64})
    center = region.center
    radius = region.radius

    n = length(center)
    integral = 0.0
    for bits in Iterators.product((0:1 for _ in 1:n)...)
        vertex = [center[i] + (2*bits[i]-1)*radius[i] for i in 1:n]
        sign = (-1)^sum(bits)
        integral += sign * antideriv(vertex)
    end

    return integral
end

function density(x_eval::Vector{Float64}, t_eval::Float64, vol_poly::SpatioTemporalPoly)
    subst = Dict(vol_poly.x_vars[i] => x_eval[i] for i in 1:length(vol_poly.x_vars))
    merge!(subst, Dict(vol_poly.t_var => t_eval))
    return subs(vol_poly.p, subst...)
end

function euler_density(x_eval::Vector{Float64}, t_eval::Float64, model::SystemModel, n_timesteps::Int=100)
    log_density = 0.0
    Δt = t_eval / n_timesteps
    div_model = divergence(model.x_vars, -model.f)
    for i in 1:n_timesteps
        x_eval -= Δt * model(x_eval)
        log_density += Δt * convert(Float64, subs(div_model, Tuple(model.x_vars) => Tuple(x_eval)))
    end 
    return exp(log_density)
end
end

function probability(region::Hyperrectangle{Float64}, integ_polynomial::SpatioTemporalPoly)
end
@polyvar x[1:2]
@polyvar t

f1 = (x[1] * (x[1] - 1)) * (-x[1]^2 + 3.0 * x[1]*x[2]^2)
f2 = (x[2] * (x[2] - 1)) * (x[1] - 4 * x[2]^2 * x[1])
#f1 = (x[1]^2 + x[1]*x[2]^2)
#f2 = (x[1] + 2 * x[2] * x[1])

model = Visualization.DataStructures.SystemModel(x, [f1, f2])

println("f1: ", f1)
println("f2: ", f2)

#density = euler_density([.5, .5], 1.0, model)
#println("Euler density: ", density)

p_vf = plot_2D_erf_space_vf(model, n_points=20)
p_erf = plot_2D_erf_space_pdf(model, 1.0, n_points=30, n_timesteps=50)
p_ss = plot_2D_pdf(model, 1.0, (-3.0, 3.0), (-3.0, 3.0), n_points=30, n_timesteps=50)
plot(p_vf, p_erf, p_ss, layout=(1,3))
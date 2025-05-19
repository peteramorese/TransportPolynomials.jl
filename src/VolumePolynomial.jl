
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
        Φ_ip1 = reynolds_operator(model.x_vars, Φ_i, model.f)
        Φ_i = Φ_ip1
        push!(coefficients, Φ_i)
    end
    return coefficients
end

function create_vol_poly(
        model::SystemModel, 
        t::Variable;
        degree::Int=1)

    coefficients = compute_coefficients(model, degree)

    t_monoms = monomials(t, 1:degree)
    taylor_scales = [1/factorial(i) for i in 1:degree]

    return (taylor_scales .* t_monoms)' * coefficients
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
    return vol_poly(x_eval, t_eval)
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

function probability(region::Hyperrectangle{Float64}, integ_polynomial::SpatioTemporalPoly)
end
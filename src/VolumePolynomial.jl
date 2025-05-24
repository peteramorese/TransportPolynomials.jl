function divergence(x::Vector{<:MultivariatePolynomials.AbstractVariable}, p::Vector{<:MultivariatePolynomials.AbstractPolynomialLike})
    divergence_poly = 0
    for i in 1:length(x)
        divergence_poly += differentiate(p[i], x[i])
    end
    return divergence_poly
end

function reynolds_operator(x::Vector{<:MultivariatePolynomials.AbstractVariable}, Φ::AbstractPolynomialLike, v::Vector{<:MultivariatePolynomials.AbstractPolynomialLike})
    Φ_scaled_field = Φ .* v
    return divergence(x, Φ_scaled_field)
end

function compute_coefficients(model::SystemModel, degree::Int=1)
    Φ_i = monomials(model.x_vars, 0)[1]
    coefficients = Vector{typeof(model.f[1])}()
    for i in 1:degree
        Φ_ip1 = reynolds_operator(model.x_vars, Φ_i, -model.f)
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

    vol_poly = (t_monoms .* taylor_scales)' * coefficients + 1
    return SpatioTemporalPoly(model.x_vars, t, vol_poly)
end

function create_vol_and_sos_bound_poly(
        model::SystemModel, 
        t::Variable,
        bounding_region::Hyperrectangle{Float64};
        degree::Int=1,
        lagrangian_degree_inc::Int=1)

    deriv_coeffs = compute_coefficients(model, degree + 1)

    t_monoms = monomials(t, 1:degree)
    taylor_scales = [1/factorial(i) for i in 1:degree]

    vol_poly = (t_monoms .* taylor_scales)' * deriv_coeffs[1:end-1] + 1

    kp1_deriv = deriv_coeffs[end]

    ## DEBUG
    #test = plot_polynomial_surface(kp1_deriv, model.x_vars[1], model.x_vars[2], (0,1), (0,1))
    #display(test)
    ##f = plot(test)
    ##display(f)
    ##

    kp1_deriv_deg = maxdegree(kp1_deriv)
    lagrangian_degree = kp1_deriv_deg + lagrangian_degree_inc
    #lagrangian_degree = 12
    @info "Degree of bound polynomial coeff: $kp1_deriv_deg, using lagrangian degree: $lagrangian_degree"

    pre_scale = 1.0
    l = sos_bound(kp1_deriv, model.x_vars, bounding_region, lagrangian_degree, upper_bound=false, pre_scale=pre_scale, silent=false)
    println("M lower bound: ", l)
    u = sos_bound(kp1_deriv, model.x_vars, bounding_region, lagrangian_degree, upper_bound=true, pre_scale=pre_scale, silent=false)
    println("M upper bound: ", u)

    kp1_deriv_bound = max(abs(l), abs(u))
    bound_poly = kp1_deriv_bound * t^(degree + 1) / factorial(degree + 1)

    return SpatioTemporalPoly(model.x_vars, t, vol_poly), TemporalPoly(t, bound_poly)
end

function create_vol_and_sos_bound_poly(
        model::SystemModel, 
        t::Variable;
        degree::Int=1,
        lagrangian_degree_inc::Int=2)
    dim = dimension(model)
    bounding_region = Hyperrectangle(low=zeros(dim), high=ones(dim))
    return create_vol_and_sos_bound_poly(model, t, bounding_region, degree=degree, lagrangian_degree_inc=lagrangian_degree_inc)
end

function create_vol_and_coeff_bound_poly(
        model::SystemModel, 
        t::Variable;
        degree::Int=1)

    deriv_coeffs = compute_coefficients(model, degree + 1)

    t_monoms = monomials(t, 1:degree)
    taylor_scales = [1/factorial(i) for i in 1:degree]

    vol_poly = (t_monoms .* taylor_scales)' * deriv_coeffs[1:end-1] + 1

    kp1_deriv = deriv_coeffs[end]

    kp1_deriv_bound = 0
    for c in coefficients(kp1_deriv)
        kp1_deriv_bound = max(kp1_deriv_bound, abs(c))
    end
    println("M coeff bound: ", kp1_deriv_bound)

    bound_poly = kp1_deriv_bound * t^(degree + 1) / factorial(degree + 1)

    return SpatioTemporalPoly(model.x_vars, t, vol_poly), TemporalPoly(t, bound_poly)
end

function create_integrator_poly(vol_poly::SpatioTemporalPoly)
    # Create the antiderivative polynomial
    p_antideriv = vol_poly.p
    for x_var in vol_poly.x_vars
        p_antideriv = antidifferentiate(p_antideriv, x_var)
    end

    return SpatioTemporalPoly(vol_poly.x_vars, vol_poly.t_var, p_antideriv)
end

#function vol_poly_bound(vol_poly::SpatioTemporalPoly)
#end

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

function propagate_sample(x_eval::Vector{Float64}, t_duration::Float64, model::SystemModel, n_timesteps::Int=100; forward::Bool=true)
    Δt = t_duration / n_timesteps

    multiplier = 1 
    if !forward 
        multiplier = -1
    end

    for i in 1:n_timesteps
        x_eval += multiplier * Δt * model(x_eval)
    end 
    return x_eval
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

function probability(region::Hyperrectangle{Float64}, t_eval::Float64, integ_polynomial::SpatioTemporalPoly)
    function antideriv(x::Vector{Float64})
        return integ_polynomial(x, t_eval)   
    end
    return evaluate_integral(antideriv, region)
end

function mc_euler_probability(region::Hyperrectangle{Float64}, model::SystemModel, t_eval::Float64, n_timesteps::Int=100, n_samples::Int=1000)
    count = 0
    for _ in 1:n_samples
        x_eval_0 = rand(Uniform(0, 1), length(model.x_vars))
        x_eval_t = propagate_sample(x_eval_0, t_eval, model, n_timesteps, forward=true)
        if x_eval_t ∈ region
            count += 1
        end
    end
    return count / n_samples
end
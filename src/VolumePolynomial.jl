function divergence(x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}, p::Vector{<:MultivariatePolynomials.AbstractPolynomialLike})
    divergence_poly = 0
    for i in 1:length(x_vars)
        divergence_poly += MultivariatePolynomials.differentiate(p[i], x_vars[i])
    end
    return divergence_poly
end

function reynolds_operator(x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}, Φ::AbstractPolynomialLike, v::Vector{<:MultivariatePolynomials.AbstractPolynomialLike})
    Φ_scaled_field = Φ .* v
    return divergence(x_vars, Φ_scaled_field)
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

function create_vol_poly(model::SystemModel, t_var::MultivariatePolynomials.AbstractVariable, degree::Int=1)

    coefficients = compute_coefficients(model, degree)

    t_monoms = monomials(t_var, 1:degree)
    taylor_scales = [1/factorial(i) for i in 1:degree]

    vol_poly = (t_monoms .* taylor_scales)' * coefficients + 1
    return SpatioTemporalPoly(model.x_vars, t_var, vol_poly)
end

function create_vol_poly_and_nxt_coeff(model::SystemModel, t_var::MultivariatePolynomials.AbstractVariable, degree::Int)

    deriv_coeffs = compute_coefficients(model, degree + 1)

    t_monoms = monomials(t_var, 1:degree)
    taylor_scales = [1/factorial(i) for i in 1:degree]

    vol_poly = (t_monoms .* taylor_scales)' * deriv_coeffs[1:end-1] + 1

    return SpatioTemporalPoly(model.x_vars, t_var, vol_poly), deriv_coeffs[end]
end

function create_integrator_poly(vol_poly::SpatioTemporalPoly)
    # Create the antiderivative polynomial
    p_antideriv = vol_poly.p
    for x_var in vol_poly.x_vars
        p_antideriv = antidifferentiate(p_antideriv, x_var)
    end

    return SpatioTemporalPoly(vol_poly.x_vars, vol_poly.t_var, p_antideriv)
end

function create_basic_sos_bound_poly(nxt_coeff::AbstractPolynomialLike, t_var::MultivariatePolynomials.AbstractVariable, deg_vol_poly::Int; lagrangian_degree_inc::Int=1, bound_type::BoundType=Magnitude)
    # Bound the nxt coefficient spatial polynomial over the whole domain (0, 1)^d
    # The integral over any region in the domain is then (naively) bounded by Vol((0, 1)^d) * M = 1 * M
    M = coeff_sos_bound(nxt_coeff::AbstractPolynomialLike, lagrangian_degree_inc=lagrangian_degree_inc, bound_type=bound_type)

    bound_poly_coeff = M / factorial(deg_vol_poly + 1)
    return TemporalPoly(t_var, polynomial(bound_poly_coeff * t_var^(deg_vol_poly + 1)))
end

function create_basic_intarith_bound_poly(nxt_coeff::AbstractPolynomialLike, t_var::MultivariatePolynomials.AbstractVariable, deg_vol_poly::Int; bound_type::BoundType=Magnitude)
    # Bound the nxt coefficient spatial polynomial over the whole domain (0, 1)^d
    # The integral over any region in the domain is then (naively) bounded by Vol((0, 1)^d) * M = 1 * M
    M = coeff_intarith_bound(nxt_coeff::AbstractPolynomialLike, bound_type=bound_type)

    bound_poly_coeff = M / factorial(deg_vol_poly + 1)
    return TemporalPoly(t_var, polynomial(bound_poly_coeff * t_var^(deg_vol_poly + 1)))
end
function divergence(p_vec::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    @assert length(p_vec) == D
    diffs = differentiate.(p_vec, 1:D)
    return reduce(add, diffs)
end

function reynolds_operator(Φ::AbstractPolynomialLike, v::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    Φ_scaled_field = Φ .* v
    #Φ_scaled_field = [product(Φ, vi) for vi in v]
    return divergence(Φ_scaled_field)
end

function compute_coefficients(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int=1) where {T, D}
    Φ_i = BernsteinPolynomial{T, D}(ones(Float64, ntuple(_ -> 1, D)...))
    coefficients = Vector{BernsteinPolynomial{T, D}}()
    push!(coefficients, Φ_i)
    for i in 1:degree
        Φ_ip1 = reynolds_operator(Φ_i, -model.f)
        Φ_i = Φ_ip1
        push!(coefficients, Φ_i)
    end
    return coefficients
end

function create_vol_poly(model::SystemModel, degree::Int=1)
    coefficients = compute_coefficients(model, degree)
    taylor_scales = [1/factorial(i) for i in 0:degree]
    return SpatioTemporalPoly(coefficients, degree, taylor_scales)
end

function create_vol_poly_and_nxt_coeff(model::SystemModel, t_var::MultivariatePolynomials.AbstractVariable, degree::Int)
    coefficients = compute_coefficients(model, degree + 1)
    taylor_scales = [1/factorial(i) for i in 0:degree]
    return SpatioTemporalPoly(coefficients[1:end-1], degree, taylor_scales), coefficients[end]
end

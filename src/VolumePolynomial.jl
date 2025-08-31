function divergence(p_vec::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    @assert length(p_vec) == D
    diffs = differentiate.(p_vec, 1:D)
    return reduce(add, diffs)
end

function reynolds_operator(Φ::BernsteinPolynomial{T,D}, v::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    #Φ_scaled_field = Φ .* v
    Φ_scaled_field = [Φ*vi for vi in v]
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

function create_vol_poly_and_nxt_coeff(model::SystemModel, degree::Int)
    coefficients = compute_coefficients(model, degree + 1)
    taylor_scales = [1/factorial(i) for i in 0:degree]
    return SpatioTemporalPoly(coefficients[1:end-1], degree, taylor_scales), coefficients[end]
end

function create_integ_poly(vol_poly::SpatioTemporalPoly, region::Hyperrectangle{Float64})
    integ_coeffs = [integrate(coeff, region) for coeff in vol_poly.spatio_coeffs]
    return TemporalPoly(vol_poly.t_deg, integ_coeffs .* vol_poly.t_coeffs)
end

function create_bound_poly(degree::Int, next_coeff::BernsteinPolynomial{T, D}) where {T, D}
    ub = upper_bound(next_coeff)
    bound_poly_coeffs = zeros(Float64, degree + 2) # Add one since the bound poly is degree + 1
    bound_poly_coeffs[end] = ub / factorial(degree + 1)
    return TemporalPoly(degree + 1, bound_poly_coeffs)
end

function create_bound_poly(degree::Int, next_coeff::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}
    ub = upper_bound(next_coeff, region)
    bound_poly_coeffs = zeros(Float64, degree + 2) # Add one since the bound poly is degree + 1
    bound_poly_coeffs[end] = ub / factorial(degree + 1)
    return TemporalPoly(degree + 1, bound_poly_coeffs)
end

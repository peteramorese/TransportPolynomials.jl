function lie_derivative(p_vec::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    dpdt_vec = similar(p_vec)
    for i in 1:D
        p_i = p_vec[i]

        spatio_derivs = [differentiate(p_i, j) for j in 1:D]

        dpdt_vec[i] = reduce(add, spatio_derivs)
    end
    return dpdt_vec
end

function compute_coefficient_vectors(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int=1) where {T, D}
    coefficients = Matrix{BernsteinPolynomial{T, D}}(undef, D, degree + 1) 
    for i in 1:degree
        coefficients[:, i + 1] = lie_derivative(coefficients[:, i])
    end
    return coefficients
end

function create_sol_poly(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int=1) where {T, D}
    coefficients = compute_coefficient_vectors(model, degree)
    taylor_scales = [1/factorial(i) for i in 0:degree]
    return FieldTemporalPoly(coefficients, degree, taylor_scales)
end

function create_sol_poly_and_nxt_coeff_vec(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int=1) where {T, D}
    coefficients = compute_coefficient_vectors(model, degree + 1)
    taylor_scales = [1/factorial(i) for i in 0:degree]
    return FieldTemporalPoly(coefficients, degree, taylor_scales), coefficients[:, end]
end

function create_field_bound_polys(degree::Int, next_coeff_vec::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    ub = upper_bound.(next_coeff_vec)
    field_bound_polys = []
    for i in 1:D
        bound_poly_coeffs = zeros(FLoat64, degree + 2)
        bound_poly_coeffs[end] = ub[i] / factorial(degree + 1)
        push!(field_bound_polys, TemporalPoly(degree + 1, bound_poly_coeffs))
    end
    return field_bound_polys
end

function taylor_expansion_basis(degree::Int)
    return TemporalPoly(degree, [1.0 / factorial(i) for i in 0:degree])
end
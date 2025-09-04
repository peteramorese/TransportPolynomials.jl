function lie_derivative(p_field::Vector{BernsteinPolynomial{T, D}}, time_diff_field::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    dpdt_vec = similar(p_field)
    for i in 1:D
        p_i = p_field[i]

        spatio_derivs = [differentiate(p_i, j) for j in 1:D]

        # Take the dot product between the gradient of the field component and hte time derivative field
        #println("spatio max: ", maximum(spatio_derivs[1].coeffs))
        #println("spatio max: ", maximum(spatio_derivs[2].coeffs))

        #for j in 1:D
        #    println("spatio i partial j: ", j, " = ", spatio_derivs[j].coeffs)
        #    println("time diff field j: ", time_diff_field[j].coeffs)
        #    println("prod: ", (spatio_derivs[j] * time_diff_field[j]).coeffs)
        #end


        dpdt_vec[i] = reduce(add, spatio_derivs .* time_diff_field)
    end
    return dpdt_vec
end

function compute_coefficient_vectors(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int=1) where {T, D}
    coefficients = ones(BernsteinPolynomial{T, D}, D, degree + 1) #Matrix{BernsteinPolynomial{T, D}}(undef, D, degree + 1) 
    #coefficients[:, 1] = model.f
    for i in 1:D
        #coeffs = zeros()
        shape = ones(Int, D)
        shape[i] = 2
        coefficients[i, 1] = BernsteinPolynomial{T, D}(reshape([0, 1], shape...))
    end
    for i in 1:(degree)
        #### TEST
        #output = lie_derivative(coefficients[:, i], model.f)
        #println("----- i+1: ")
        #println("output[1]: ", output[1].coeffs)
        #readline()
        #### 

        coefficients[:, i + 1] = lie_derivative(coefficients[:, i], model.f)
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
    return FieldTemporalPoly(coefficients[:, 1:(end-1)], degree, taylor_scales), coefficients[:, end]
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

function create_field_bound_polys(degree::Int, next_coeff_vec::Vector{BernsteinPolynomial{T, D}}, region::Hyperrectangle{Float64}) where {T, D}
    ub = upper_bound.(next_coeff_vec, region)
    field_bound_polys = []
    for i in 1:D
        bound_poly_coeffs = zeros(FLoat64, degree + 2)
        bound_poly_coeffs[end] = ub[i] / factorial(degree + 1)
        push!(field_bound_polys, TemporalPoly(degree + 1, bound_poly_coeffs))
    end
    return field_bound_polys
end

function create_bernstein_expansion(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int, duration::Float64=1.0, deg_incr::Int=0) where {T, D}
    println("deg: ", degree)
    sol_poly, nxt_coeff_vec = create_sol_poly_and_nxt_coeff_vec(model, degree)
    #println(maximum(sol_poly.spatio_vector_field_coeffs[1, end].coeffs))
    println("t coeffs: ", sol_poly.t_coeffs)
    #println("sol poly f1: ", sol_poly.spatio_vector_field_coeffs[1, 2].coeffs)
    taylor_scaled_coeffs = [sol_poly.t_coeffs[i] * sol_poly.spatio_vector_field_coeffs[:, i] for i in 1:(degree + 1)] # Converts into scaled vector of vectors

    # Append the bound poly
    #println("ub: ", nxt_coeff_vec[1].coeffs)
    ub_mag = max.(abs.(upper_bound.(nxt_coeff_vec)), abs.(lower_bound.(nxt_coeff_vec)))
    #ub = upper_bound.(nxt_coeff_vec)
    #println("upper bound: ", ub)
    println("upper bound mag: ", ub_mag)
    bound_spatio_vector_field = ones(BernsteinPolynomial{T, D}, D) .* ub_mag / factorial(degree + 1)

    push!(taylor_scaled_coeffs, bound_spatio_vector_field)

    last_ts_coeffs = taylor_scaled_coeffs[end]
    for i in 1:D
        println(" BOUND COEFFS: ", last_ts_coeffs[i].coeffs)
    end

    # Bernsteinify the scaled coefficients to get a Bernstein polynomial in space and time
    #println(maximum(taylor_scaled_coeffs[end][1].coeffs))
    println("sp vf coeffs size: ", size(sol_poly.spatio_vector_field_coeffs))
    println("len ts coefs:", length(taylor_scaled_coeffs))
    bernstein_time_coeffs = bernsteinify(taylor_scaled_coeffs, deg_incr)

    for ts_coefs in taylor_scaled_coeffs
        println("ts-")
        for c in ts_coefs
            println(size(c.coeffs))
        end
    end
    for time_coeff in bernstein_time_coeffs
        println("-")
        for c in time_coeff
            println(size(c.coeffs))
        end
    end
    #println(maximum(bernstein_time_coeffs[end][1].coeffs))

    # Convert to proper Bernstein polynomial type
    bernstein_expansions = Vector{BernsteinPolynomial{T, D+1}}(undef, D)
    for i in 1:D
        max_size = zeros(Int, D)
        for j in 1:(degree + 1)
            max_size = max.(max_size, size(bernstein_time_coeffs[j][i].coeffs))
        end
        for j in 1:(degree + 1)
            bernstein_time_coeffs[j][i] = increase_degree(bernstein_time_coeffs[j][i], tuple((max_size .- 1)...))
        end

        #println("spatio size: ", size(sol_poly.spatio_vector_field_coeffs[i, end].coeffs))
        #println("time coeff size: ", size(bernstein_time_coeffs[end][i].coeffs))
        shape = tuple(degree + 2, max_size...) # Augment with time dimension
        coeffs = Array{T, D+1}(undef, shape)
        for j in 1:(degree + 2)
            coeffs[j, (Colon() for _ in 1:D)...] = bernstein_time_coeffs[j][i].coeffs
        end
        
        bernstein_expansions[i] = BernsteinPolynomial{T, D+1}(coeffs)
    end
    return bernstein_expansions
end

function bernsteinify(coefficients::Vector{Vector{BernsteinPolynomial{T, D}}}, deg_incr::Int=0) where {T, D}
    n = length(coefficients) - 1 # Original degree of monomial basis
    m = n + deg_incr # Degree of Bernstein representation
    
    #lg = [lgamma(i) for i in 0:(m+1)]

    #bern_coeffs = zeros(eltype(coefficients), m + 1) 
    bern_coeffs = [zeros(BernsteinPolynomial{T, D}, D) for _ in 1:(m+1)]
    for i in 0:m
        for k in 0:min(i, n)
            #bern_coeffs[i+1] += coefficients[k + 1] * binomial(i, k) / binomial(m, k)
            bern_coeffs[i+1] = bern_coeffs[i+1] .+ coefficients[k + 1] * binomial(i, k) / binomial(m, k)
            #println("k+1: ", k+1, " af shape: ", size(bern_coeffs[i+1][1].coeffs), " coeffs size: ", size(coefficients[k + 1][1].coeffs))
            #println("new bernie: ", bern_coeffs[i+1][1].coeffs)
            #readline()
        end
    end
    return bern_coeffs
end
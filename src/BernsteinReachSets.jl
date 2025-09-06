"""
Create a Bernstein-Taylor expansion about a region
"""
struct BernsteinFieldExpansion{T, tD}
    field_expansion_lb::Vector{BernsteinPolynomial{T, tD}}
    field_expansion_ub::Vector{BernsteinPolynomial{T, tD}}
    duration::Float64
    region::Hyperrectangle
end

struct TransitionSet
    set::Hyperrectangle
    duration::Float64
end

struct Flowpipe
    transition_sets::Vector{TransitionSet}
end

function rescale_duration(bfe::BernsteinFieldExpansion{T, tD}, new_duration::Float64) where {T, tD}
    D = tD -1
    new_lb = Vector{BernsteinPolynomial{T, tD}}(undef, D)
    new_ub = Vector{BernsteinPolynomial{T, tD}}(undef, D)
    for i in 1:D
        new_lb[i] = affine_transform(bfe.field_expansion_lb[i], dim=1, lower=0.0, upper=new_duration)
        new_ub[i] = affine_transform(bfe.field_expansion_ub[i], dim=1, lower=0.0, upper=new_duration)
    end
    return BernsteinFieldExpansion{T, tD}(new_lb, new_ub, new_duration, bfe.region)
end

function reposition(bfe::BernsteinFieldExpansion{T, tD}, new_region::Hyperrectangle) where {T, tD}
    D = tD -1
    new_lb = Vector{BernsteinPolynomial{T, tD}}(undef, D)
    new_ub = Vector{BernsteinPolynomial{T, tD}}(undef, D)

    mins = low(new_region)
    maxes = high(new_region)

    for i in 1:D
        for i in 1:D
            new_lb[i] = affine_transform(bfe.field_expansion_lb[i], dim=(i+1), lower=mins[i], upper=maxes[i])
            new_ub[i] = affine_transform(bfe.field_expansion_ub[i], dim=(i+1), lower=mins[i], upper=maxes[i])
        end
    end
    return BernsteinFieldExpansion{T, tD}(new_lb, new_ub, bfe.duration, new_region)
end

function get_final_region(bfe::BernsteinFieldExpansion{T, tD}, t::Float64=nothing) where {T, tD}
    D = tD -1
    t_eval = isnothing(t) ? bfe.duration : t
    mins = []
    maxes = []
    for i in 1:D
        # Evaluate the Bernstein-Taylor expansions at the desired time
        space_bern_lb = decasteljau(bfe.field_expansion_lb[i], dim=1, xi=t_eval)
        space_bern_ub = decasteljau(bfe.field_expansion_ub[i], dim=1, xi=t_eval)

        push!(mins, lower_bound(space_bern_lb))
        push!(maxes, upper_bound(space_bern_ub))
    end
    return Hyperrectangle(low=mins, high=maxes)
end

function get_transition_region(bfe::BernsteinFieldExpansion{T, tD}) where {T, tD}
    D = tD -1
    mins = []
    maxes = []
    for i in 1:D
        # Bound the expansion over the whole region and time interval
        push!(mins, lower_bound(bfe.field_expansion_lb[i]))
        push!(maxes, upper_bound(bfe.field_expansion_ub[i]))
    end
    return TransitionSet(Hyperrectangle(low=mins, high=maxes), bfe.duration)
end

function compute_bernstein_reach_sets(model::SystemModel{BernsteinPolynomial{T, D}}, init_set::Hyperrectangle, duration::Float64; expansion_degree::Int=4, Δt_max::Float64=1.0) where {T, D}
    # Create the original field expansion
    bfe = create_bernstein_field_expansion(model, expansion_degree, duration=Δt_max)

    start_set = init_set 
    total_time = 0.0

    trans_sets = Vector{TransitionSet}()
    while total_time < duration
        #println("bfe region: ", low(bfe.region), " - ", high(bfe.region))
        bfe_start_set = reposition(bfe, start_set)
        trans_region = get_transition_region(bfe_start_set)
        end_set = get_final_region(bfe_start_set, trans_region.duration)
        println("total_time: ", total_time)
        println("start set region: ", low(start_set), " - ", high(start_set))
        println("end set region: ", low(end_set), " - ", high(end_set))
        println("trn set region: ", low(trans_region.set), " - ", high(trans_region.set))

        # Clamp set to domain
        end_set_clamp_low = max.(low(end_set), zeros(Float64, D))
        end_set_clamp_high = min.(high(end_set), ones(Float64, D))
        end_set = Hyperrectangle(low=end_set_clamp_low, high=end_set_clamp_high)

        push!(trans_sets, trans_region)
        start_set = end_set
        total_time += trans_region.duration
    end

    return Flowpipe(trans_sets)
end

function lie_derivative(p_field::Vector{BernsteinPolynomial{T, D}}, time_diff_field::Vector{BernsteinPolynomial{T, D}}) where {T, D}
    dpdt_vec = similar(p_field)
    for i in 1:D
        p_i = p_field[i]

        spatio_derivs = [differentiate(p_i, j) for j in 1:D]

        dpdt_vec[i] = reduce(add, spatio_derivs .* time_diff_field)
    end
    return dpdt_vec
end

function compute_coefficient_vectors(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int=1) where {T, D}
    coefficients = ones(BernsteinPolynomial{T, D}, D, degree + 1) #Matrix{BernsteinPolynomial{T, D}}(undef, D, degree + 1) 
    for i in 1:D
        shape = ones(Int, D)
        shape[i] = 2
        coefficients[i, 1] = BernsteinPolynomial{T, D}(reshape([0, 1], shape...))
    end
    for i in 1:(degree)
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

function create_bernstein_expansion(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int; duration::Float64=1.0, deg_incr::Int=0, upper::Bool=true) where {T, D}
    sol_poly, nxt_coeff_vec = create_sol_poly_and_nxt_coeff_vec(model, degree)
    
    # Reparameterize time to the duration
    duration_scaled_t_coeffs = [duration^((i-1)) * sol_poly.t_coeffs[i] for i in 1:length(sol_poly.t_coeffs)]
    taylor_scaled_coeffs = [duration_scaled_t_coeffs[i] * sol_poly.spatio_vector_field_coeffs[:, i] for i in 1:(degree + 1)] # Converts into scaled vector of vectors

    # Append the bound poly
    #ub_mag = max.(abs.(upper_bound.(nxt_coeff_vec)), abs.(lower_bound.(nxt_coeff_vec)))
    if upper
        ub = upper_bound.(nxt_coeff_vec)
        bound_spatio_vector_field = ones(BernsteinPolynomial{T, D}, D) .* ub * duration^(degree + 1) / factorial(degree + 1)
    else
        lb = lower_bound.(nxt_coeff_vec)
        bound_spatio_vector_field = ones(BernsteinPolynomial{T, D}, D) .* lb * duration^(degree + 1) / factorial(degree + 1)
    end
    #println("upper bound: ", ub)
    #println("upper bound mag: ", ub_mag)

    push!(taylor_scaled_coeffs, bound_spatio_vector_field)

    # Bernsteinify the scaled coefficients to get a Bernstein polynomial in space and time
    bernstein_time_coeffs = bernsteinify(taylor_scaled_coeffs, deg_incr)


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

        shape = tuple(degree + 2, max_size...) # Augment with time dimension
        coeffs = Array{T, D+1}(undef, shape)
        for j in 1:(degree + 2)
            coeffs[j, (Colon() for _ in 1:D)...] = bernstein_time_coeffs[j][i].coeffs
        end
        
        bernstein_expansions[i] = BernsteinPolynomial{T, D+1}(coeffs)
    end
    return bernstein_expansions
end

function create_bernstein_field_expansion(model::SystemModel{BernsteinPolynomial{T, D}}, degree::Int; duration::Float64=1.0, deg_incr::Int=0) where {T, D}
    lb = create_bernstein_expansion(model, degree, duration=duration, deg_incr=deg_incr, upper=false)
    ub = create_bernstein_expansion(model, degree, duration=duration, deg_incr=deg_incr, upper=true)
    region = Hyperrectangle(low=zeros(D), high=ones(D))
    return BernsteinFieldExpansion(lb, ub, duration, region)
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

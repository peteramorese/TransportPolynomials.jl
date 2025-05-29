function sos_bound(p::AbstractPolynomialLike, vars::Vector{<:MultivariatePolynomials.AbstractVariable}, region::Hyperrectangle{Float64}, lagrangian_degree::Int; upper_bound::Bool=false, pre_scale::Float64=1.0, silent=true)
    d = length(vars)
    
    if pre_scale != 1.0
        # Scale the polynomial and the region
        p = scale_polynomial(p, vars, pre_scale)
        region = scale_region(region, pre_scale)
    end
    #println("Bounding: ", p, " \n in region: ", region)

    if maxdegree(p) > lagrangian_degree
        @warn "Polynomial degree ($(maxdegree(p))) exceeds the specified Lagrangian degree ($lagrangian_degree). Consider increasing Lagrangian degree."
    end

    q = upper_bound ? -p : p  # Flip sign if maximizing

    model = SOSModel( optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 1e-10, "MSK_IPAR_NUM_THREADS" => 1))
    if silent
        set_silent(model)  # Optional: suppress solver output
    end

    # SoS variable gamma (the bound)
    @variable(model, γ)

    # Define constraints of [0,1]^d
    g = [polynomial(x - l) for (x, l) in zip(vars, low(region))]                 # x_i - l ≥ 0
    append!(g, [polynomial(u - x) for (x, u) in zip(vars, high(region))])        # u - x_i ≥ 0

    # SoS polynomial σ₀ (free term)
    monobasis = monomials(vars, 0:lagrangian_degree)
    @variable(model, σ₀, SOSPoly(monomials(vars, 0:lagrangian_degree)))

    # Multiplier SoS polynomials σᵢ for each gᵢ
    @variable(model, σ[1:length(g)], SOSPoly(monomials(vars, 0:lagrangian_degree)))

    # Construct the decomposition constraint:
    #     q(x) - γ == σ₀(x) + ∑ σᵢ(x) * gᵢ(x)
    decomposition = σ₀
    for (gi, σi) in zip(g, σ)
        decomposition += σi * gi
    end

    #@constraint(model, q - γ - decomposition >= 0, basis = ChebyshevBasisFirstKind)
    @constraint(model, q - γ - decomposition >= 0)

    @objective(model, Max, γ)

    optimize!(model)

    #println("Termination_status: ", termination_status(model))
    if termination_status(model) != MOI.OPTIMAL
        @warn "SoS optimization did not converge to optimality" termination_status(model)
    end

    bound = value(γ)
    return upper_bound ? -bound : bound
end

function scale_region(region::Hyperrectangle{Float64}, scale::Float64)
    # Scale the center and radius of the region
    scaled_center = region.center * scale
    scaled_radius = region.radius * scale

    return Hyperrectangle(scaled_center, scaled_radius)
end     

function scale_polynomial(p::AbstractPolynomialLike, vars::Vector{<:MultivariatePolynomials.AbstractVariable}, scale::Float64)
    subst = Dict(x => (1/scale) * x for x in vars)
    return subs(p, subst...)
end

function sos_coeff_mag_bound(coeff::AbstractPolynomialLike; lagrangian_degree_inc::Int=1, bounding_region::Union{Hyperrectangle{Float64}, Nothing}=nothing)
    if isnothing(bounding_region)
        dim = nvariables(coeff)
        bounding_region = Hyperrectangle(low=zeros(dim), high=ones(dim))
    end
        
    coeff_deg = maxdegree(coeff)
    lagrangian_degree = coeff_deg + lagrangian_degree_inc
    #lagrangian_degree = 12
    @info "SoS Bounding coefficient polynomial of degree : $coeff_deg, using lagrangian degree: $lagrangian_degree"

    pre_scale = 1.0
    l = sos_bound(coeff, variables(coeff), bounding_region, lagrangian_degree, upper_bound=false, pre_scale=pre_scale, silent=false)
    println("M lower bound: ", l)
    u = sos_bound(coeff, variables(coeff), bounding_region, lagrangian_degree, upper_bound=true, pre_scale=pre_scale, silent=false)
    println("M upper bound: ", u)

    return max(abs(l), abs(u))
end

function reduce_poly(p::AbstractPolynomialLike, max_degree::Int, upper_bound::Bool=false)
    
end
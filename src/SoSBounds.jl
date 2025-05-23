function sos_bound(p::AbstractPolynomialLike, vars::Vector{<:MultivariatePolynomials.AbstractVariable}, region::Hyperrectangle{Float64}, lagrangian_degree::Int; upper_bound::Bool=false)
    d = length(vars)
    q = upper_bound ? -p : p  # Flip sign if maximizing

    model = SOSModel(Mosek.Optimizer)
    set_silent(model)  # Optional: suppress solver output

    # SoS variable gamma (the bound)
    @variable(model, γ)

    # Define constraints of [0,1]^d
    g = [polynomial(v) for v in vars]                 # x_i ≥ 0
    println("type of g: ", typeof(g))
    test = [1 - v for v in vars]
    println("test type: ", typeof(test))
    append!(g, [polynomial(1 - v) for v in vars])     # 1 - x_i ≥ 0

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

    @constraint(model, q - γ - decomposition >= 0)

    @objective(model, Max, γ)

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        error("SoS optimization did not converge to optimality")
    end

    bound = value(γ)
    return upper_bound ? -bound : bound
end
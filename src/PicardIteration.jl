
function picard_operator(model::SystemModel, stpv::SpatioTemporalPolyVector, truncation_degree::Int)
    function dim_operator(dynamics_dim_poly::AbstractPolynomialLike)
        subst_composition = Dict(model.x_vars[i] => stpv.p[i] for i in 1:dimension(stpv))
        composed_poly = subs(dynamics_dim_poly, subst_composition...)
        while maxdegree(composed_poly) > truncation_degree
            composed_poly = remove_leading_term(composed_poly)
        end
        integral_poly = antidifferentiate(composed_poly, stpv.t_var)
        return integral_poly
    end
    picard_result = dim_operator.(model.f) + polynomial.(stpv.x_vars)
    return picard_result
end

function picard_vol_poly(model::SystemModel, t_var::MultivariatePolynomials.AbstractVariable, iterations::Int, truncation_degree::Int)
    @polyvar vol_var
    augmented_state_vars = [model.x_vars; vol_var]

    volume_ode = divergence(model.x_vars, -model.f) * vol_var
    augmented_ode = [-model.f; volume_ode]

    augmented_system = SystemModel(augmented_state_vars, augmented_ode)
    
    # Constant initial picard function
    initial_picard_poly = polynomial.(augmented_state_vars)

    stpv = SpatioTemporalPolyVector(augmented_state_vars, t_var, initial_picard_poly)

    for i in 1:iterations
        println("Iteration: ", i)
        stpv.p = picard_operator(augmented_system, stpv, truncation_degree)
    end

    # Remove the volume variable by setting it to 1 (the initial volume)

    vol_poly = stpv.p[end]
    vol_poly = subs(vol_poly, vol_var => 1.0)
    return SpatioTemporalPoly(model.x_vars, t_var, vol_poly)
end
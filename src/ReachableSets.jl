# Compute reachability using TMJets from a SystemModel
function compute_taylor_reach_sets(model::SystemModel; init_set::Hyperrectangle, duration::Float64)
    n = dimension(model)

    #poly_funcs = [polynomial_function]
    function f!(dx, x, p, t)
        for i in 1:n
            dx[i] = model.f[i](model.x_vars => x)
        end
        return dx
    end

    #sys = PolynomialContinuousSystem(model.f, dimension(model))
    sys = RA.BlackBoxContinuousSystem(f!, dimension(model))

    prob = RA.InitialValueProblem(sys, init_set)

    # Run reachability using TMJets
    sol = RA.solve(prob; T=duration, alg=RA.TMJets(), verbose=false)

    return sol
end

# Compute final overapproximating hyperrectangle
function compute_final_hyperrectangle(sol::RA.ReachSolution)
    final_set = sol[end]
    return RA.set(RA.overapproximate(final_set, Hyperrectangle))
end

function propagate_set(model::SystemModel; init_set::Hyperrectangle, duration::Float64)
    sol = compute_taylor_reach_sets(model; init_set=init_set, duration=duration)
    return compute_final_hyperrectangle(sol)
end
# Compute reachability using TMJets from a SystemModel
function compute_taylor_reach_sets(model::SystemModel; init_set::Hyperrectangle, duration::Float64)
    n = dimension(model)

    #sys = PolynomialContinuousSystem(model.f, dimension(model))

    #poly_funcs = [polynomial_function]
    function f!(dx, x, p, t)
        for i in 1:n
            dx[i] = model.f[i](model.x_vars => x)
        end
        return dx
    end

    #sys = PolynomialContinuousSystem(model.f, dimension(model))
    sys = BlackBoxContinuousSystem(f!, dimension(model))

    prob = InitialValueProblem(sys, init_set)
    #fn_name = gensym(:system_vector_field!)
    #@eval @taylorize function $system_vector_field!(du, u, p, t)
    #    du = $model(u)
    #    return du
    #end

    #println("init set: ", init_set)
    #expr = :(@ivp(x' = 2*x, dim:$n, x(0) ∈ $init_set))
    #prob = @eval @ivp(x' = $fn_name(x), dim:$n, x(0) ∈ $init_set)
    #prob = @eval @ivp(x' = $model(x), dim:$n, x(0) ∈ $init_set)
    #prob = eval(expr)

    # Run reachability using TMJets
    sol = solve(prob; T=duration, alg=TMJets(), verbose=false)

    return sol
end

# Compute final overapproximating hyperrectangle
function compute_final_hyperrectangle(sol::ReachabilityAnalysis.ReachSolution)
    final_set = sol[end]
    return set(overapproximate(final_set, Hyperrectangle))
end
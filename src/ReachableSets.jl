#function to_state_space_model(model::SystemModel{BernsteinPolynomial{T, D}}) where {T, D}
#end

function to_mv_polynomial_system(model::SystemModel{BernsteinPolynomial{T, D}}, x_vars::Vector) where {T, D}
    @assert length(x_vars) == dimension(model) "Number of variables must match polynomial dimension."
    mvp_f = to_mv_polynomial.(model.f, Ref(x_vars))
    return SystemModel(mvp_f)
end

function compute_taylor_reach_sets(model::SystemModel; init_set::Hyperrectangle, duration::Float64) 

    function f!(dx, x, p, t)
        for i in 1:D
            dx[i] = model.f[i](x)
            #dx[i] = decasteljau(model.f[i], x)[1]
        end
        return dx
    end

    D = dimension(model)

    domain = Hyperrectangle(low=zeros(D), high=ones(D))


    println("D: ", D)
    #sys = RA.ConstrainedPolynomialContinuousSystem(model.f, domain)
    #sys = RA.PolynomialContinuousSystem(model.f)
    #sys = PolynomialContinuousSystem(model.f, dimension(model))

    sys = RA.BlackBoxContinuousSystem(f!, dimension(model))
    #sys = RA.BlackBoxContinuousSystem(f!, dimension(model), dom=domain, rem=rem)

    prob = RA.InitialValueProblem(sys, init_set)

    # Run reachability using TMJets
    println("Creating flowpipe...")
    sol = RA.solve(prob; T=duration, alg=RA.TMJets(), verbose=false)
    println(" Done.")

    #print(typeof(sol))
    return RA.flowpipe(sol)
end

function compute_hyperrectangle(sol::RA.AbstractFlowpipe, idx::Int)
    return RA.set(RA.overapproximate(sol[idx], Hyperrectangle))
end

function compute_hyperrectangle(reach_set::RA.AbstractReachSet, idx::Int)
    return RA.set(RA.overapproximate(reach_set, Hyperrectangle))
end

#function propagate_set(model::SystemModel; init_set::Hyperrectangle, duration::Float64)
#    sol = compute_taylor_reach_sets(model; init_set=init_set, duration=duration)
#    return compute_final_hyperrectangle(sol)
#end
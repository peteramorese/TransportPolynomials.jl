function to_mv_polynomial_system(model::SystemModel{BernsteinPolynomial{T, D}}, x_vars::Vector) where {T, D}
    @assert length(x_vars) == dimension(model) "Number of variables must match polynomial dimension."
    mvp_f = to_mv_polynomial.(model.f, Ref(x_vars))
    return SystemModel(mvp_f)
end

function compute_taylor_reach_sets(model::SystemModel; init_set::Hyperrectangle, duration::Float64) 

    function f!(dx, x, p, t)
        for i in 1:D
            #dx[i] = model.f[i](x)
            dx[i] = model.f[i](x)
            #dx[i] = decasteljau(model.f[i], x)[1]
        end
        #dx = decasteljau.(model.f, Ref(x))
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

    return RA.flowpipe(sol)
end

function compute_hyperrectangle(sol::RA.AbstractFlowpipe, idx::Int)
    return RA.set(RA.overapproximate(sol[idx], Hyperrectangle))
end

function compute_hyperrectangle(reach_set::RA.AbstractReachSet, idx::Int)
    return RA.set(RA.overapproximate(reach_set, Hyperrectangle))
end

function x_flowpipe_to_u_flowpipe(dtf::DistributionTransform{DIST, D}, fp::RA.AbstractFlowpipe) where {DIST, D}
    u_sets = [Rx_to_Ru(dtf, compute_hyperrectangle(fp, i)) for i in 1:length(fp)]
    return RA.DiscreteFlowpipe(u_sets, fp.t_vec)
end
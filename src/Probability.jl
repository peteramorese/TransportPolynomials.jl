function density(x_eval::Vector{Float64}, t_eval::Float64, vol_poly::SpatioTemporalPoly)
    return vol_poly(x_eval, t_eval)
end

function probability(region::Hyperrectangle{Float64}, t_eval::Float64, integ_poly::SpatioTemporalPoly)
    function antideriv(x::Vector{Float64})
        return integ_poly(x, t_eval)   
    end
    return evaluate_integral(antideriv, region)
end

function probability_and_ubound(region::Hyperrectangle{Float64}, t_eval::Float64, integ_poly::SpatioTemporalPoly, bound_poly::TemporalPoly)
    prob = probability(region, t_eval, integ_poly)
    bound = min(1.0, prob + bound_poly(t_eval))
    return prob, bound
end

function probability_and_geometric_ubound(region::Hyperrectangle{Float64}, t_eval::Float64, integ_poly::SpatioTemporalPoly, bound_poly::TemporalPoly)
    prob = probability(region, t_eval, integ_poly)
    taylor_bound = bound_poly(t_eval)
    if taylor_bound < 1.0
        bound = prob / (1 - taylor_bound)
    else
        bound = 1.0
    end
    return prob, bound
end

function evaluate_integral(antideriv, region::Hyperrectangle{Float64})
    center = region.center
    radius = region.radius

    n = length(center)
    integral = 0.0
    for bits in Iterators.product((0:1 for _ in 1:n)...)
        vertex = [center[i] + (2*bits[i]-1)*radius[i] for i in 1:n]
        sign = (-1)^sum(bits)
        integral += sign * antideriv(vertex)
    end

    return integral
end

function propagate_sample(x_eval::Vector{Float64}, t_duration::Float64, model::SystemModel; n_timesteps::Int=100, forward::Bool=true)
    Δt = t_duration / n_timesteps

    multiplier = 1 
    if !forward 
        multiplier = -1
    end

    for i in 1:n_timesteps
        x_eval += multiplier * Δt * model(x_eval)
    end 
    return x_eval
end

function euler_density(x_eval::Vector{Float64}, t_eval::Float64, model::SystemModel; n_timesteps::Int=100)
    log_density = 0.0
    Δt = t_eval / n_timesteps
    div_model = divergence(model.x_vars, -model.f)
    for i in 1:n_timesteps
        x_eval -= Δt * model(x_eval)
        log_density += Δt * convert(Float64, subs(div_model, Tuple(model.x_vars) => Tuple(x_eval)))
    end 
    return exp(log_density)
end

function mc_euler_probability(region::Hyperrectangle{Float64}, t_eval::Float64, model::SystemModel; n_timesteps::Int=100, n_samples::Int=1000)
    count = 0
    for _ in 1:n_samples
        x_eval_0 = rand(Uniform(0, 1), length(model.x_vars))
        x_eval_t = propagate_sample(x_eval_0, t_eval, model, n_timesteps=n_timesteps, forward=true)
        if x_eval_t ∈ region
            count += 1
        end
    end
    return count / n_samples
end
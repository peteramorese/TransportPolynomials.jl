function euler_density(x_eval::Vector{Float64}, duration::Float64; forward_model::SystemModel{BernsteinPolynomial{T, D}}, n_timesteps::Int=100) where {T, D}
    log_density = 0.0
    Δt = duration / n_timesteps
    div_model = divergence(-forward_model.f)
    for i in 1:n_timesteps
        x_eval -= Δt * model(x_eval)
        log_density += Δt * div_model.(x_eval)
    end 
    return exp(log_density)
end

function euler_probability(region::Hyperrectangle{Float64}, duration::Float64; forward_model::SystemModel, n_timesteps::Int=100, n_samples::Int=1000) 
    count = 0
    for _ in 1:n_samples
        x_eval_0 = rand(D)
        x_eval_t = propagate_sample(x_eval_0, duration, forward_model, n_timesteps=n_timesteps, forward=true)
        if x_eval_t ∈ region
            count += 1
        end
    end
    return count / n_samples
end

function euler_probability_traj(region::Hyperrectangle{Float64}, duration::Float64; forward_model::SystemModel, n_timesteps::Int=100, n_samples::Int=1000, confidence::Float64=0.99, sampler=nothing) 
    D = dimension(forward_model)
    count = zeros(n_timesteps)
    for _ in 1:n_samples
        if isnothing(sampler)
            x_eval_0 = rand(D)
        else
            x_eval_0 = sampler()
        end
        x_eval_t_traj = propagate_sample_traj(x_eval_0, duration, forward_model, n_timesteps=n_timesteps, forward=true)
        for k in 1:n_timesteps
            x_eval_t = x_eval_t_traj[k, :]
            if x_eval_t ∈ region
                count[k] += 1
            end
        end
    end
    means = count / n_samples
    hoeffding_bounds = hoeffding_bound(n_samples, confidence) * ones(n_timesteps)
    bernstein_bounds = bernstein_bound.(Ref(n_samples), Ref(confidence), means)
    return means, range(0.0, duration, n_timesteps), hoeffding_bounds, bernstein_bounds
end

function avg_fwd_state(duration::Float64; foward_model::SystemModel, n_timesteps::Int=100, n_samples::Int=1000)
    
end

"""
Sample random points uniformly from a hyperrectangle. Returns (n_samples, D) matrix
"""
function sample_region(region::Hyperrectangle{Float64}, n_samples::Int)
    lower_bounds = low(region)
    upper_bounds = high(region)
    return lower_bounds' .+ rand(n_samples, LazySets.dim(region)) .* (upper_bounds - lower_bounds)'
end

function hoeffding_bound(n_samples::Int, confidence::Float64)
    return sqrt(log(2/confidence)/(2*n_samples))
end

function bernstein_bound(n_samples::Int, confidence::Float64, empirical_mean::Float64)
    v = empirical_mean * (1 - empirical_mean)                # empirical variance for Bernoulli
    log_term = log(2 / confidence)
    a = n_samples / 3                      # coefficient for ε² term
    b = n_samples * v                      # coefficient for ε term
    c = log_term / 2                       # constant term
    return (-b + sqrt(b^2 + 4*a*c)) / (2*a)
end
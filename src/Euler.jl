"""
Compute the final desination of a state subject to a field
"""
function propagate_sample(x_eval::Vector{Float64}, duration::Float64, model::SystemModel; n_timesteps::Int=100, forward::Bool=true)
    Δt = duration / n_timesteps

    multiplier = 1 
    if !forward 
        multiplier = -1
    end

    for i in 1:n_timesteps
        x_eval += multiplier * Δt * model(x_eval)
    end 
    return x_eval
end

"""
Return the Euler-integration trajectory of a sample
"""
function propagate_sample_traj(x_eval::Vector{Float64}, duration::Float64, model::SystemModel; n_timesteps::Int=100, forward::Bool=true)
    Δt = duration / n_timesteps

    multiplier = 1 
    if !forward 
        multiplier = -1
    end

    x_traj = Matrix{Float64}(undef, n_timesteps + 1, length(x_eval))
    x_traj[1, :] = x_eval
    for i in 1:n_timesteps
        x_eval += multiplier * Δt * model(x_eval)
        x_traj[i+1, :] = x_eval
    end 
    return x_traj
end

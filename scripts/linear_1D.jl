using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
using LinearAlgebra
plotly()
#pyplot()

# Specifications #
a = -0.5
true_system, dtf = linear_system_1D(a=a)
target_region = Hyperrectangle(low=[-0.3], high=[0.2])
duration = 2.2
model_degrees = 40 * ones(Int, dimension(true_system))
fp_deg = 5
vp_deg = 5 # Volume polynomial degree
deg_incr = 0
Δt_max = 0.02
λ = 30.0
##################

function eval_lin_sys(A::Matrix{Float64}, init_mean::Vector{Float64}, init_cov::Matrix{Float64}, t::Float64, region::Hyperrectangle{Float64})
    # Compute matrix exponential for linear system propagation
    exp_At = exp(A * t)
    
    # Propagate the mean: μ(t) = exp(At) * μ(0)
    propagated_mean = exp_At * init_mean
    
    # Propagate the covariance: Σ(t) = exp(At) * Σ(0) * exp(At)ᵀ
    # Since A and init_cov are diagonal, this simplifies significantly
    propagated_cov = exp_At * init_cov * exp_At'
    
    # For diagonal matrices, we can compute the closed-form integral
    # The integral of a multivariate Gaussian over a hyperrectangle
    # can be computed as the product of 1D Gaussian CDF differences
    
    # Extract region bounds
    low_bounds = low(region)
    high_bounds = high(region)
    
    # For diagonal covariance, the integral factorizes
    integral_value = 1.0
    for i in 1:length(propagated_mean)
        # Standardize the bounds
        std_dev = sqrt(propagated_cov[i, i])
        if std_dev > 0
            integral_value *= (cdf(Normal(propagated_mean[i], std_dev), high_bounds[i]) - cdf(Normal(propagated_mean[i], std_dev), low_bounds[i]))
        else
            # If std_dev = 0, check if mean is in the region
            if low_bounds[i] <= propagated_mean[i] <= high_bounds[i]
                integral_value *= 1.0
            else
                integral_value *= 0.0
            end
        end
    end
    
    return integral_value
end

X, fx_hat = generate_data(true_system, 10000; domain_std=2.5, noise_std=0.00)

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# System regression
learned_rmodel = constrained_system_regression(U, fu_hat, model_degrees, reverse=true, λ=λ)

#target_region_u = Rx_to_Ru(dtf, target_region)
#println("Vol U target region: ", volume(target_region_u))
#
#flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=fp_deg, Δt_max=Δt_max, deg_incr=deg_incr)
#
#box_ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
#tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
#geo_ts = create_geometric_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
#
## Compute monte carlo eval
#euler_prob_traj, timestamps = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=1000, n_timesteps=100)
#function sampler()
#    return [rand(Normal(0.0, 0.4)) for i in 1:1]
#end
#euler_prob_traj_true, timestamps_true = euler_probability_traj(target_region, duration, forward_model=true_system, n_samples=1000, n_timesteps=100, sampler=sampler)
#
### Plot the flowpipe
##plt_fp = plot()
##plt_fp = plot_flowpipe!(plt_fp, flow_pipe; color=:teal, alpha=0.0)
##plt_fp = plot_2D_region!(plt_fp, target_region_u; color=:red)
#
## Plot the probability functions
#plt_prob = plot()
#plt_prob = plot_taylor_spline!(plt_prob, box_ts, duration, label="Box TS", color=:blue)
#plt_prob = plot_taylor_spline!(plt_prob, tamed_ts, duration, label="Tamed TS", color=:coral)
#plt_prob = plot_taylor_spline!(plt_prob, geo_ts, duration, label="Geo TS", color=:green)
#plt_prob = plot(plt_prob, timestamps, euler_prob_traj, label="MC (learned)", color=:red)
#hline!(plt_prob, [0.0, 1.0], linestyle=:dash, color=:black, label=nothing)
#
## Plot the ground truth probability function
#truth_prob = [eval_lin_sys(a, [0.0], diagm([0.4].^2), t, target_region) for t in timestamps]
#plt_prob = plot(plt_prob, timestamps, truth_prob, label="Ground Truth", color=:black)
#plt_prob = plot(plt_prob, timestamps_true, euler_prob_traj_true, label="MC (ground truth)", color=:purple)

# Plot the vector fields for comparison
#x_vals = range(-0.3, 0.2, length=100)
#learned_model_x = to_state_space_model(dtf, -learned_rmodel)
#vf_learned_vals = [learned_model_x([x])[1] for x in x_vals]
#vf_true_vals = [true_system([x])[1] for x in x_vals]

x_vals = range(0.0, 1.0, length=100)
true_model_u = to_u_space_model(dtf, true_system)
vf_learned_vals = [-learned_rmodel([x])[1] for x in x_vals]
vf_true_vals = [true_model_u([x])[1] for x in x_vals]
println("x vals size: ", size(x_vals), " vf learned vals size: ", size(vf_learned_vals), " vf true vals size: ", size(vf_true_vals))

plt_vf_learned = plot()
plt_vf_true = plot()
plt_vf_learned = plot(x_vals, vf_learned_vals, label="Learned", color=:blue)
plt_vf_true = plot(x_vals, vf_true_vals, label="True", color=:black)


#plot_vector_field!(plt_vf_learned, learned_model_x)
#plot_vector_field!(plt_vf_true, true_system)

# Plot the comparison
#plt_comp = plot(plt_fp, plt_prob, plt_vf_learned, plt_vf_true, layout=(1,4), size=(1200, 400))
plt_comp = plot(plt_vf_learned, plt_vf_true, layout=(1,2), size=(1200, 400))
plot(plt_comp)
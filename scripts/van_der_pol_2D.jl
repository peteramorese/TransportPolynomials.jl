using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
#plotly()
pyplot()

# Specifications #
true_system, dtf = van_der_pol(μ=1.0)
target_region = Hyperrectangle(low=[0.0, 0.0], high=[0.3, 0.3])
duration = 5.0
model_degrees = 7 * ones(Int, dimension(true_system))
fp_deg = 4
vp_deg = 5 # Volume polynomial degree
deg_incr = 0
Δt_max = 0.1#0.01
λ = 10.0
save_plots = true
# CHANGE MC SAMPLES BACK TO 10000
##################


X, fx_hat = generate_data(true_system, 2000; domain_std=0.5, noise_std=0.01)

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# System regression
learned_rmodel = constrained_system_regression(U, fu_hat, model_degrees, reverse=true, λ=λ)

target_region_u = Rx_to_Ru(dtf, target_region)

flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=fp_deg, Δt_max=Δt_max, deg_incr=deg_incr)

box_ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
geo_ts = create_geometric_taylor_spline(flow_pipe, learned_rmodel, vp_deg)

# Compute monte carlo eval
euler_prob_traj, timestamps, hoeffding_bounds, bernstein_bounds = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=1000, n_timesteps=100)
function sampler()
    return [rand(Normal(0.0, 0.5)) for i in 1:2]
end
euler_prob_traj_true, timestamps_true, hoeffding_bounds_true, bernstein_bounds_true = euler_probability_traj(target_region, duration, forward_model=true_system, n_samples=1000, n_timesteps=100, sampler=sampler)

euler_prob_upper = [p + b for (p, b) in zip(euler_prob_traj, hoeffding_bounds)]
euler_prob_upper_true = [p + b for (p, b) in zip(euler_prob_traj_true, hoeffding_bounds_true)]

# Plot the flowpipe
plt_fp = plot(aspect_ratio = :equal)
plt_fp = plot_flowpipe!(plt_fp, flow_pipe; color=:teal, alpha=0.0, n=1)
plt_fp = plot_2D_region!(plt_fp, target_region_u; color=:red)

# Plot the probability functions
plt_prob = plot(legend=:topleft, tickfontsize=12, legendfontsize=12)  
plt_prob = plot_taylor_spline!(plt_prob, box_ts, duration, label="Box", color=:blue)
plt_prob = plot_taylor_spline!(plt_prob, tamed_ts, duration, label="Bound 1", color=:coral)
plt_prob = plot_taylor_spline!(plt_prob, geo_ts, duration, label="Bound 2 (geometric)", color=:green)
plt_prob = plot(plt_prob, timestamps, euler_prob_traj, label="MC (learned)", color=:red)

plt_prob = plot(plt_prob, timestamps_true, euler_prob_traj_true, label="MC (ground truth)", color=:purple)

xlabel!(plt_prob, "τ", fontsize=12)
xlims!(plt_prob, 0.0, duration)
ylims!(plt_prob, 0.0, 1.0)

plot(plt_fp, plt_prob, layout=(1,3), size=(1200, 400))

# Save plots if flag is set
if save_plots
    figures_dir = joinpath(@__DIR__, "..", "figures")
    mkpath(figures_dir)  # Create directory if it doesn't exist
    
    # Save flowpipe plot as PNG
    #savefig(plt_fp, joinpath(figures_dir, "van_der_pol_2D_flowpipe.pdf"))
    savefig(plt_fp, joinpath(figures_dir, "van_der_pol_2D_flowpipe.png"))
    
    # Save probability plot as PDF
    savefig(plt_prob, joinpath(figures_dir, "van_der_pol_2D_probability.pdf"))
    
    println("Plots saved to figures directory")
end
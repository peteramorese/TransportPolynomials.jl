using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
using Profile
using LoopVectorization
#plotly()
pyplot()


# Specifications #
true_system, dtf = cartpole(l=1.0, mc=2.0)
#target_region = Hyperrectangle(low=[-0.2, -0.1, -0.1, 0.1], high=[0.2, 0.1, 0.1, 0.4])
target_region = Hyperrectangle(low=[-0.1, -0.1, -0.1, 0.1], high=[0.0, 0., 0., 0.2])
duration = 2.5
model_degrees = 6 * ones(Int, dimension(true_system))
fp_deg = 3
vp_deg = 4 # Volume polynomial degree
deg_incr = 10
Δt_max = 0.01
save_plots = true
##################


X, fx_hat = generate_data(true_system, 20000; domain_std=0.5, noise_std=0.01)

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

target_region_u = Rx_to_Ru(dtf, target_region)
println("Target region in U: ", low(target_region_u), " - ", high(target_region_u), " radius: ", radius_hyperrectangle(target_region_u))

# System regression
println("Regressing model...")
learned_rmodel = constrained_system_regression(U, fu_hat, model_degrees, reverse=true, λ=20.0)
println("Done!")

max_coeffs = [maximum(abs.(fi.coeffs)) for fi in learned_rmodel.f]
println("Maximum coeff magnitude: ", maximum(max_coeffs))

println("Computing reachable sets...")
flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=fp_deg, Δt_max=Δt_max, deg_incr=deg_incr)
println("Done!")

println("Creating box taylor spline...")
box_ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
println("Creating tamed taylor spline...")
tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
println("Creating geometric taylor spline...")
geo_ts = create_geometric_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
println("Done!")

# Compute monte carlo eval
euler_prob_traj, timestamps, hoeffding_bounds, bernstein_bounds = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=10000, n_timesteps=100)
function sampler()
    return [rand(Normal(0.0, 0.5)), 
        rand(Normal(0.0, 0.1)),
        rand(Normal(0.1, 0.2)),
        rand(Normal(0.5, 0.4))]
end
euler_prob_traj_true, timestamps_true, hoeffding_bounds_true, bernstein_bounds_true = euler_probability_traj(target_region, duration, forward_model=true_system, n_samples=10000, n_timesteps=100, sampler=sampler)

# Compute upper bounds (mean + bernstein bound) in log space
euler_prob_upper_log = [log(max(p + b, 1e-10)) for (p, b) in zip(euler_prob_traj, hoeffding_bounds)]
euler_prob_upper_log_true = [log(max(p + b, 1e-10)) for (p, b) in zip(euler_prob_traj_true, hoeffding_bounds_true)]

# Plot the flowpipe
plt_fp = plot(aspect_ratio = :equal)
plt_fp = plot_flowpipe!(plt_fp, flow_pipe; color=:teal, alpha=0.0, n=10)
plt_fp = plot_2D_region!(plt_fp, target_region_u; color=:red, alpha=0.8)

# Plot the probability functions (log scale)
plt_prob = plot()
# Compute log probabilities for Taylor splines
n_pts = 100
t_pts = range(0.0, duration, length=n_pts)
tamed_ts_pts = [tamed_ts(t) for t in t_pts]
box_ts_pts = [box_ts(t) for t in t_pts]
geo_ts_pts = [geo_ts(t) for t in t_pts]
log_box_ts_pts = [log(max(b, 1e-10)) for b in box_ts_pts]
log_tamed_ts_pts = [log(max(t, 1e-10)) for t in tamed_ts_pts]
log_geo_ts_pts = [log(max(g, 1e-10)) for g in geo_ts_pts]
euler_prob_traj_log = [log(max(p, 1e-10)) for p in euler_prob_traj]
euler_prob_traj_log_true = [log(max(p, 1e-10)) for p in euler_prob_traj_true]

plt_prob = plot(legend=false, tickfontsize=12)  
plt_prob = plot!(plt_prob, t_pts, log_box_ts_pts, label="Box", color=:blue)
plt_prob = plot!(plt_prob, t_pts, log_tamed_ts_pts, label="Bound 1", color=:coral)
plt_prob = plot!(plt_prob, t_pts, log_geo_ts_pts, label="Bound 2 (geometric)", color=:green)
plt_prob = plot!(plt_prob, timestamps, euler_prob_traj_log, label="MC (learned)", color=:red)
plt_prob = plot!(plt_prob, timestamps_true, euler_prob_traj_log_true, label="MC (ground truth)", color=:purple)

xlabel!(plt_prob, "τ", fontsize=12)
xlims!(plt_prob, 0.0, duration)
ylims!(plt_prob, -10.0, 0.0)

plot(plt_fp, plt_prob, layout=(1,3), size=(1200, 400))

# Save plots if flag is set
if save_plots
    figures_dir = joinpath(@__DIR__, "..", "figures")
    mkpath(figures_dir)  # Create directory if it doesn't exist
    
    # Save flowpipe plot as PNG
    #savefig(plt_fp, joinpath(figures_dir, "cartpole_4D_rare_event_flowpipe.pdf"))
    savefig(plt_fp, joinpath(figures_dir, "cartpole_4D_rare_event_flowpipe.png"))
    
    # Save probability plot as PDF
    savefig(plt_prob, joinpath(figures_dir, "cartpole_4D_rare_event_probability.pdf"))
    
    println("Plots saved to figures directory")
end
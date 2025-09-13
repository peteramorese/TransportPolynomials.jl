using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
plotly()
#pyplot()

# Specifications
true_system, dtf = van_der_pol(μ=1.0)
target_region = Hyperrectangle(low=[0.0, 0.0], high=[0.4, 0.4])
duration = 1.15
vp_deg = 5 # Volume polynomial degree


X, fx_hat = generate_data(true_system, 2000; domain_std=0.5, noise_std=0.01)


U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# Plot the data
plt_x_data = quiver(X[:, 1], X[:, 2], quiver=(fx_hat[:, 1], fx_hat[:, 2]), title="X space data")
plt_u_data = quiver(U[:, 1], U[:, 2], quiver=(fu_hat[:, 1], fu_hat[:, 2]), title="U space data")
plot(plt_x_data, plt_u_data, layout=(1, 2))

# System regression
learned_rmodel = constrained_system_regression(U, fu_hat, [5, 5], reverse=true)

target_region_u = Rx_to_Ru(dtf, target_region)

deg_incr = 20
Δt_max = 0.1
expansion_deg = 5
flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=expansion_deg, Δt_max=Δt_max, deg_incr=deg_incr)

box_ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)

# Compute monte carlo eval
euler_prob_traj, timestamps = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=1000, n_timesteps=4)

# Plot the flowpipe
plt_fp = plot()
plt_fp = plot_flowpipe!(plt_fp, flow_pipe; color=:teal, alpha=0.0)
plt_fp = plot_2D_region!(plt_fp, target_region_u; color=:red)

# Plot the probability functions
plt_prob = plot()
plt_prob = plot_taylor_spline!(plt_prob, box_ts, duration, label="Box TS", color=:blue)
plt_prob = plot_taylor_spline!(plt_prob, tamed_ts, duration, label="Tamed TS", color=:coral)
plt_prob = plot(plt_prob, timestamps, euler_prob_traj, label="MC (learned)", color=:red)
hline!(plt_prob, [0.0, 1.0], linestyle=:dash, color=:black, label=nothing)

plot(plt_fp, plt_prob, layout=(1,3), size=(1200, 400))
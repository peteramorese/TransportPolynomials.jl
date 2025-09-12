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
true_system, dtf = cartpole()
target_region = Hyperrectangle(low=[0.2, 0.2], high=[0.4, 0.4])
duration = 0.45
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

plt_fp = plot_2D_flowpipe(flow_pipe)
plt_fp = plot_2D_region(plt_fp, target_region_u)

ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)

# Sample trajectories from the system to validate the reach sets
n_val_traj = 20
lower_bounds = low(target_region_u)
upper_bounds = high(target_region_u)

x_trajs = []
x_traj_init_states = lower_bounds' .+ rand(n_val_traj, 2) .* (upper_bounds - lower_bounds)'
x_traj_init_states[end, :] = [.5564253, .5628805]
for i in 1:n_val_traj
    x_traj = propagate_sample_traj(x_traj_init_states[i, :], duration, learned_rmodel, n_timesteps=500)
    push!(x_trajs, x_traj)
end

# Plot the probability functions
n_pts = 100
plt_vp_prob = plot()
t_pts = range(0.0, duration, length=n_pts)
ts_pts = [ts(t) for t in t_pts]
tamed_ts_pts = [tamed_ts(t) for t in t_pts]
plot!(plt_vp_prob, t_pts, ts_pts, label="Box Taylor Spline")
plot!(plt_vp_prob, t_pts, tamed_ts_pts, label="Tamed Taylor Spline")
hline!(plt_vp_prob, [0.0, 1.0], linestyle=:dash)

for x_traj in x_trajs
    plot!(plt_fp, x_traj[:, 1], x_traj[:, 2], label=nothing)
    plot!(plt_end_sets, x_traj[:, 1], x_traj[:, 2], label=nothing)
end


plot(plt_fp, plt_end_sets, plt_vp_prob, layout=(1,3), size=(1200, 400), legend=false)
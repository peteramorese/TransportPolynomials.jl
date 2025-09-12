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

# Port the model to x space for flowpipe computation
#learned_rmodel_x = to_state_space_model(dtf, learned_rmodel)
target_region_u = Rx_to_Ru(dtf, target_region)

### TEST
#target_region_u = Hyperrectangle(low=[0.5564252697664893, 0.5628805173789195], high=[0.7863510709052677, 0.7904654471020098])
#target_region_u = Hyperrectangle(low=[0.5558629902178214, 0.5624853257907159], high=[0.7862882641569429, 0.7904350215573519])
#duration = 0.0501
### 

#println("target region: ", target_region_u)
#println("init set region: ", low(target_region_u), " - ", high(target_region_u))

#flow_pipe = compute_taylor_reach_sets(learned_rmodel; init_set=target_region_u, duration=duration, eval_fcn=log_eval)
#plot(flow_pipe, vars=(1,2))

deg_incr = 20
Δt_max = 0.1
expansion_deg = 5
println("FLOW PIPE")
flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=expansion_deg, Δt_max=Δt_max, deg_incr=deg_incr)
println("FLOW PIPE END SETS")
flow_pipe_end_sets = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=expansion_deg, Δt_max=Δt_max, deg_incr=deg_incr, return_transition_sets=false)

#println("TRANSITION SETS:")
#for ts in flow_pipe.transition_sets
#    println(" - " , low(ts.set), " - ", high(ts.set))
#end
#println("penultimate trn set - " , low(flow_pipe.transition_sets[end-1].set), " - ", high(flow_pipe.transition_sets[end-1].set))
#println("penultimate end set - " , low(flow_pipe_end_sets.transition_sets[end-1].set), " - ", high(flow_pipe_end_sets.transition_sets[end-1].set))
#println("   ultimate trn set - " , low(flow_pipe.transition_sets[end].set), " - ", high(flow_pipe.transition_sets[end].set))
#println("   ultimate end set - " , low(flow_pipe_end_sets.transition_sets[end].set), " - ", high(flow_pipe_end_sets.transition_sets[end].set))

plt_fp = plot_2D_flowpipe(flow_pipe)
plt_fp = plot_2D_region(plt_fp, target_region_u)
plt_end_sets = plot_2D_flowpipe(flow_pipe_end_sets)
plt_end_sets = plot_2D_region(plt_end_sets, target_region_u)
#display(plt_fp)

ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)


## Plot the bounding boxes of each segment
#plt_boxes = plot()
#xlims!(plt_boxes, 0.0, 1.0)
#ylims!(plt_boxes, 0.0, 1.0)
#for segment in tamed_ts.segments
#    plot_2D_region(plt_boxes, segment.Ω_bounding_box)
#end

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

#u1_ls = range(0.01, 0.99, length=20)
#u2_ls = range(0.01, 0.99, length=20)
#u1_grid = repeat(u1_ls, 1, length(u2_ls)) |> vec
#u2_grid = repeat(transpose(u2_ls), length(u1_ls), 1) |> vec
#U_grid = hcat(u1_grid, u2_grid)
#f_learned_grid = learned_rmodel(U_grid)
#
## Compare with ground truth
#true_system_u = to_u_space_model(dtf, true_system)
#f_true_grid = true_system_u(U_grid)
#
#
#plt_learned_vf = quiver(u1_grid, u2_grid, quiver=(f_learned_grid[:, 1], f_learned_grid[:, 2]), title="Learned U space vector field")
#xlims!(plt_learned_vf, (-0.1, 1.1))
#ylims!(plt_learned_vf, (-0.1, 1.1))
#plt_true_vf = quiver(u1_grid, u2_grid, quiver=(f_true_grid[:, 1], f_true_grid[:, 2]), title="True U space vector field")
#xlims!(plt_true_vf, (-0.1, 1.1))
#ylims!(plt_true_vf, (-0.1, 1.1))
#plt_data_pts = scatter(U[:, 1], U[:, 2], title="U data pts") # Show the data sparsity
#xlims!(plt_data_pts, (-0.1, 1.1))
#ylims!(plt_data_pts, (-0.1, 1.1))
#
#plot(plt_learned_vf, plt_true_vf, plt_data_pts, plt_fp, layout=(1, 4))




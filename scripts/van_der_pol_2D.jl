using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
pyplot()

# Specifications
true_system, dtf = van_der_pol(μ=1.0)
target_region = Hyperrectangle(low=[0.2, 0.2], high=[0.201, 0.201])
duration = 1.2


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

#println("target region: ", target_region_u)
println("start set region: ", low(target_region_u), " - ", high(target_region_u))

flow_pipe = compute_taylor_reach_sets(learned_rmodel; init_set=target_region_u, duration=duration, eval_fcn=log_eval)
plot(flow_pipe, vars=(1,2))

#flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, 1.2, expansion_degree=6, Δt_max=0.05)
#plt_fp = plot_2D_flowpipe(flow_pipe)
#display(plt_fp)


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




using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
pyplot()


μ = 1.0
true_system = SystemModel([
    x -> x[2],
    x -> μ * (1.0 .- x[1].^2) .* x[2] .- x[1] 
])

X, fx_hat = generate_data(true_system, 2000; domain_std=0.5, noise_std=0.01)

dtf_dists = [Normal(0.0, 0.5), Normal(0.0, 0.5)] # Specifies the initial distribution
dtf = DistributionTransform(SVector(dtf_dists...))

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# Plot the data
plt_x_data = quiver(X[:, 1], X[:, 2], quiver=(fx_hat[:, 1], fx_hat[:, 2]), title="X space data")
plt_u_data = quiver(U[:, 1], U[:, 2], quiver=(fu_hat[:, 1], fu_hat[:, 2]), title="U space data")
plot(plt_x_data, plt_u_data, layout=(1, 2))

# System regression
learned_model = constrained_system_regression(U, fu_hat, [4, 4])
#println()
#println("MODEL f1: ", learned_model.f[1].coeffs)
#println("MODEL f1 deg incr: ", increase_degree(learned_model.f[1], (5,4)).coeffs)
#println()

duration = 1.0
bernstein_expansion_ub = create_bernstein_expansion(learned_model, 8, upper=true)
bernstein_expansion_lb = create_bernstein_expansion(learned_model, 8, upper=false)

u_test = [0.2, 0.3]

t_dur = 0.2
t_pts = 100
t_ls = range(0, t_dur, t_pts)
input = ones(t_pts, 3)
input[:, 1] = t_ls
input[:, 2] *= u_test[1]
input[:, 3] *= u_test[2]

D = dimension(learned_model)
plts = []

euler_vals = Matrix{Float64}(undef, D, length(t_ls))
for (k, t) in enumerate(t_ls)
    u_test_f = propagate_sample(u_test, t, learned_model, n_timesteps=100)    
    euler_vals[:, k] = u_test_f
end

for i in 1:D
    expansion_vals_ub = bernstein_expansion_ub[i](input)
    expansion_vals_lb = bernstein_expansion_lb[i](input)

    plt = plot(t_ls, euler_vals[i, :], label="euler")
    plot!(plt, t_ls, expansion_vals_ub, label="expansion")
    plot!(plt, t_ls, expansion_vals_lb, label="expansion")
    push!(plts, plt)
end
plot(plts..., layout=(D, 1))





#u1_ls = range(0.01, 0.99, length=20)
#u2_ls = range(0.01, 0.99, length=20)
#u1_grid = repeat(u1_ls, 1, length(u2_ls)) |> vec
#u2_grid = repeat(transpose(u2_ls), length(u1_ls), 1) |> vec
#U_grid = hcat(u1_grid, u2_grid)
#f_learned_grid = learned_model(U_grid)
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
#plot(plt_learned_vf, plt_true_vf, plt_data_pts, layout=(1, 3))




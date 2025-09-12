using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
plotly()


# Specifications
true_system, dtf = van_der_pol(μ=1.0)
target_region = Hyperrectangle(low=[0.2, 0.2], high=[0.3, 0.3])
#duration = 5.2
#duration = 0.5
vp_deg = 3 # Volume polynomial degree


X, fx_hat = generate_data(true_system, 2000; domain_std=0.5, noise_std=0.01)


U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# Plot the data
plt_x_data = quiver(X[:, 1], X[:, 2], quiver=(fx_hat[:, 1], fx_hat[:, 2]), title="X space data")
plt_u_data = quiver(U[:, 1], U[:, 2], quiver=(fu_hat[:, 1], fu_hat[:, 2]), title="U space data")
plot(plt_x_data, plt_u_data, layout=(1, 2))

# System regression
learned_rmodel = constrained_system_regression(U, fu_hat, [5, 5], reverse=true)

#target_region_u = Rx_to_Ru(dtf, target_region)

target_region_u = Hyperrectangle(low=[0.5564252697664893, 0.5628805173789195], high=[0.7863510709052677, 0.7904654471020098])
#target_region_u = Hyperrectangle(low=[0.5558629902178214, 0.5624853257907159], high=[0.7862882641569429, 0.7904350215573519])


#println()
#println("MODEL f1: ", learned_model.f[1].coeffs)
#println("MODEL f1 deg incr: ", increase_degree(learned_model.f[1], (5,4)).coeffs)
#println()

duration = 1.0
#duration = 2.0
#bernstein_expansion_ub = create_bernstein_expansion(learned_model, 5, upper=true, duration=duration)
#bernstein_expansion_lb = create_bernstein_expansion(learned_model, 5, upper=false, duration=duration)

test_t = 0.05
bfe = create_bernstein_field_expansion(learned_rmodel, 5, duration=test_t, deg_incr=0)
bfe_start_set = reposition(bfe, target_region_u)
#end_set = get_final_region(bfe_start_set, 0.1)
println(">>end set region: ", low(end_set), " - ", high(end_set))

#bernstein_expansion_ub = bfe.field_expansion_ub
#bernstein_expansion_lb = bfe.field_expansion_lb
#u_test = [.5564253, .5628805]

bernstein_expansion_ub = bfe_start_set.field_expansion_ub
bernstein_expansion_lb = bfe_start_set.field_expansion_lb
u_test = [0.0, 0.0]

#t_dur = 0.05
t_dur = 1.0
t_pts = 100
t_ls = range(0, t_dur, t_pts)
input = ones(t_pts, 3)
input[:, 1] = t_ls
input[:, 2] *= u_test[1]
input[:, 3] *= u_test[2]

D = dimension(learned_rmodel)
plts = []

#time_rescaled_model_f = learned_model.

euler_vals = Matrix{Float64}(undef, D, length(t_ls))
for (k, t) in enumerate(t_ls)
    u_test_f = propagate_sample(u_test, t, learned_rmodel, n_timesteps=100)    
    euler_vals[:, k] = u_test_f
end

#println("DIM: ", i, " Lower bound over t: ", lower_bound(bernstein_expansion_lb[i]))

facet_expansion_lb = decasteljau(bernstein_expansion_lb[1], dim=(1+1), xi=0.0)
println("time edge lb: ", lower_bound(facet_expansion_lb))
space_bern_lb = decasteljau(facet_expansion_lb, dim=1, xi=1.0)
println("Facet lower bound D1: ", lower_bound(space_bern_lb))

space_bern_lb = decasteljau(bernstein_expansion_lb[1], dim=1, xi=test_t)
facet_expansion_lb = decasteljau(space_bern_lb, dim=(1), xi=0.0)
println("Facet lower bound D1: ", lower_bound(facet_expansion_lb))

for i in 1:D
    expansion_vals_ub = bernstein_expansion_ub[i](input)
    expansion_vals_lb = bernstein_expansion_lb[i](input)

    println("DIM: ", i, " Lower bound over t: ", lower_bound(bernstein_expansion_lb[i]))
    #lower_bound

    #plt = plot(t_ls, euler_vals[i, :], label="euler")
    #plot!(plt, t_ls, expansion_vals_ub, label="expansion_upper")
    #plot!(plt, t_ls, expansion_vals_lb, label="expansion_lower")
    plt = plot(t_ls, euler_vals[i, :], label=nothing)
    plot!(plt, t_ls, expansion_vals_ub, label=nothing)
    plot!(plt, t_ls, expansion_vals_lb, label=nothing)
    ylims!(plt, 0.0, 1.0)
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




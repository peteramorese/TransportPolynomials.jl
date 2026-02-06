using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
using Random
pyplot()

Random.seed!(1)

#p = BernsteinPolynomial(rand(3, 3 ,3))
p = BernsteinPolynomial(rand(20, 20))

region = Hyperrectangle(low=[0.6, 0.01], high=[0.8, 0.6])
p_tf = affine_transform(p, region)

p_tf_sep = affine_transform(p, dim=1, lower=low(region)[1], upper=high(region)[1])
p_tf_sep = affine_transform(p_tf_sep, dim=1, lower=low(region)[1], upper=high(region)[1])

u1_ls = range(0.01, 0.99, length=50)
u2_ls = range(0.01, 0.99, length=50)
u1_grid = repeat(u1_ls, 1, length(u2_ls)) |> vec
u2_grid = repeat(transpose(u2_ls), length(u1_ls), 1) |> vec
U_grid = hcat(u1_grid, u2_grid)
p_grid = p(U_grid)
p_tf_grid = p_tf(U_grid)


plt_p = Plots.surface(u1_grid, u2_grid, p_grid)
plt_p_tf = Plots.surface(u1_grid, u2_grid, p_tf_grid)

plot(plt_p, plt_p_tf, layout=(1,2))


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




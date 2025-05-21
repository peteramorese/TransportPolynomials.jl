using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets
pyplot()

@polyvar x[1:2]
@polyvar t

f1 = (x[1] * (x[1] - 1)) * (-x[1]^2 + 3.0 * x[1]*x[2]^2)
f2 = (x[2] * (x[2] - 1)) * (x[1] - 4 * x[2]^2 * x[1])
#f1 = (x[1]^2 + x[1]*x[2]^2)
#f2 = (x[1] + 2 * x[2] * x[1])

model = SystemModel(x, [f1, f2])

println("f1: ", f1)
println("f2: ", f2)

#density = euler_density([.5, .5], 1.0, model)
#println("Euler density: ", density)

time = 1.0

erf_space_region = Hyperrectangle(low=[0.4, .6], high=[0.6, 1.0])

p_vf = plot_2D_erf_space_vf(model, scale=.3, n_points=20)
p_erf = plot_2D_erf_space_pdf(model, time, n_points=30, n_timesteps=50)
p_ss = plot_2D_pdf(model, time, (-3.0, 3.0), (-3.0, 3.0), n_points=30, n_timesteps=50)
plot_2D_region(p_erf, erf_space_region, alpha=0.5)
fig1 = plot(p_vf, p_erf, p_ss, layout=(1,3))

vol_poly = create_vol_poly(model, t, degree=10)
p_vp_erf = plot_2D_erf_space_pdf(vol_poly, time, n_points=30)
p_pv_ss = plot_2D_pdf(vol_poly, time, (-3.0, 3.0), (-3.0, 3.0), n_points=30)
fig2 = plot(p_vp_erf, p_pv_ss, layout=(1,2))

display(fig1)
display(fig2)

mc_prob = mc_euler_probability(erf_space_region, model, time)
println("Monte Carlo probability: ", mc_prob)

integ_poly = create_integrator_polynomial(vol_poly)
vol_poly_prob = probability(erf_space_region, time, integ_poly)
println("Volume polynomial probability: ", vol_poly_prob)

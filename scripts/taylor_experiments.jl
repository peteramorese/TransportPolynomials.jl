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

order = 3

erf_space_region = Hyperrectangle(low=[0.4, .6], high=[0.6, 1.0])

vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, t, order)

println("Created volume polynomial")

bound_poly = create_basic_sos_bound_poly(nxt_coeff, t, order, lagrangian_degree_inc=2, upper_only=true)

println("Created bound polynomial")

integ_poly = create_integrator_poly(vol_poly)

println("Created integrator polynomial")



## Plot the pdf at a given time
# Monte Carlo
plt_vf = plot_2D_erf_space_vf(model, scale=.3, n_points=20)
plt_erf = plot_2D_erf_space_pdf(model, time, n_points=30, n_timesteps=50)
plt_ss = plot_2D_pdf(model, time, (-3.0, 3.0), (-3.0, 3.0), n_points=30, n_timesteps=50)
plot_2D_region_in_3D(plt_erf, erf_space_region, alpha=0.5)

# Vol poly
plt_vp_erf = plot_2D_erf_space_pdf(vol_poly, time, n_points=30)
plt_pv_ss = plot_2D_pdf(vol_poly, time, (-3.0, 3.0), (-3.0, 3.0), n_points=30)

fig1 = plot(plt_vf, plt_erf, plt_ss, layout=(1,3))
fig2 = plot(plt_vp_erf, plt_pv_ss, layout=(1,2))

## Plot the density prediction over time
u = [0.5, 0.2]
duration = 5.0
#plt_vp_density = plot_vol_poly_density_vs_time(u, vol_poly, bound_poly, duration, n_points=100)
plt_vp_density = plot_vol_poly_density_vs_time(u, vol_poly, duration, n_points=100)
plt_vp_density = plot_euler_density_vs_time(plt_vp_density, u, model, duration, n_points=100)
fig3 = plot(plt_vp_density, title="Density vs. time for u=$u")

## Plot the probability prediction over time
plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, bound_poly, duration, n_points=50, geometric=false)
plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, bound_poly, duration, plt=plt_vp_prob, n_points=50, geometric=true)
#plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, duration, n_points=50)
plt_vp_prob = plot_euler_mc_prob_vs_time(plt_vp_prob, erf_space_region, model, duration, n_points=50, n_samples=500)
fig4 = plot(plt_vp_prob, title="Probability vs. time for region")


## Compare integration methods
#mc_prob = mc_euler_probability(erf_space_region, time, model)
#println("Monte Carlo probability: ", mc_prob)
#
#vol_poly_prob = probability(erf_space_region, time, integ_poly)
#println("Volume polynomial probability: ", vol_poly_prob)

display(fig1)
display(fig2)
display(fig3)
display(fig4)
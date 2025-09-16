using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
plotly()
#pyplot()

# Specifications #
true_system, dtf = cartpole()
target_region = Hyperrectangle(low=[0.1, 0.05, 0.0, 0.0], high=[0.15, 0.07, 0.4, 0.4])
duration = 1.0
model_degrees = 5 * ones(Int, dimension(true_system))
fp_deg = 3
vp_deg = 3 # Volume polynomial degree
deg_incr = 0
Δt_max = 0.005
##################


X, fx_hat = generate_data(true_system, 20000; domain_std=0.5, noise_std=0.01)

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

target_region_u = Rx_to_Ru(dtf, target_region)
println("Target region in U: ", low(target_region_u), " - ", high(target_region_u))

# System regression
println("Regressing model...")
learned_rmodel = constrained_system_regression(U, fu_hat, model_degrees, reverse=true, λ=10.0)
println("Done!")

max_coeffs = [maximum(abs.(fi.coeffs)) for fi in learned_rmodel.f]
println("Maximum coeff magnitude: ", maximum(max_coeffs))

println("Computing reachable sets...")
flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=fp_deg, Δt_max=Δt_max, deg_incr=deg_incr)
println("Done!")

println("Creating box taylor spline...")
box_ts = create_box_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
print("Creating tamed taylor spline...")
tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
println("Done!")

# Compute monte carlo eval
euler_prob_traj, timestamps = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=1000, n_timesteps=100)

# Plot the flowpipe
plt_fp = plot()
plt_fp = plot_flowpipe!(plt_fp, flow_pipe; vars=(1, 3), color=:teal, alpha=0.0)
plt_fp = plot_2D_region!(plt_fp, target_region_u; vars=(1, 3), color=:red)

# Plot the probability functions
plt_prob = plot()
plt_prob = plot_taylor_spline!(plt_prob, box_ts, duration, label="Box TS", color=:blue)
plt_prob = plot_taylor_spline!(plt_prob, tamed_ts, duration, label="Tamed TS", color=:coral)
plt_prob = plot(plt_prob, timestamps, euler_prob_traj, label="MC (learned)", color=:red)
hline!(plt_prob, [0.0, 1.0], linestyle=:dash, color=:black, label=nothing)

plot(plt_fp, plt_prob, layout=(1,3), size=(1200, 400))
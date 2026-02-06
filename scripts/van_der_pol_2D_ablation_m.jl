using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using Plots
using LazySets
using Distributions
using StaticArrays
#plotly()
pyplot()

# Specifications #
true_system, dtf = van_der_pol(μ=1.0)
target_region = Hyperrectangle(low=[0.0, 0.0], high=[0.3, 0.3])
duration = 5.0
model_degrees = 6 * ones(Int, dimension(true_system))
fp_deg = 4
vp_deg_range = [1, 2, 3, 5, 7] # Volume polynomial degree range for ablation study
deg_incr = 0
Δt_max = 0.1
λ = 10.0
save_plots = true
##################


X, fx_hat = generate_data(true_system, 2000; domain_std=0.5, noise_std=0.01)

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# System regression
learned_rmodel = constrained_system_regression(U, fu_hat, model_degrees, reverse=true, λ=λ)

target_region_u = Rx_to_Ru(dtf, target_region)

flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=fp_deg, Δt_max=Δt_max, deg_incr=deg_incr)

# Compute monte carlo eval
euler_prob_traj, timestamps, hoeffding_bounds, bernstein_bounds = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=10000, n_timesteps=100)
function sampler()
    return [rand(Normal(0.0, 0.5)) for i in 1:2]
end
euler_prob_traj_true, timestamps_true, hoeffding_bounds_true, bernstein_bounds_true = euler_probability_traj(target_region, duration, forward_model=true_system, n_samples=10000, n_timesteps=100, sampler=sampler)

# Store taylor splines for each vp_deg
tamed_ts_list = []
geo_ts_list = []

# Compute tamed and geo taylor splines for each vp_deg
println("Computing taylor splines for vp_deg range: $vp_deg_range")
for vp_deg in vp_deg_range
    println("  Computing for vp_deg = $vp_deg...")
    tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
    geo_ts = create_geometric_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
    push!(tamed_ts_list, tamed_ts)
    push!(geo_ts_list, geo_ts)
end


# Plot the probability functions
plt_prob = plot(legend=:topleft, tickfontsize=12, legendfontsize=12)  

colors_by_vp_deg = Dict(
    1 => :blue,
    3 => :red,
    5 => :green,
    7 => :orange,
    2 => :purple,
    4 => :brown,
    6 => :pink,
    8 => :gray
)

# Plot all tamed taylor splines (bound 1) - solid lines
for (vp_deg, tamed_ts, geo_ts) in zip(vp_deg_range, tamed_ts_list, geo_ts_list)
    global plt_prob
    color = colors_by_vp_deg[vp_deg]
    plt_prob = plot_taylor_spline!(plt_prob, tamed_ts, duration, label="Bound 1 (m=$vp_deg)", color=color, linestyle=:solid)
    plt_prob = plot_taylor_spline!(plt_prob, geo_ts, duration, label="Bound 2 (m=$vp_deg)", color=color, linestyle=:dash)
end

xlabel!(plt_prob, "τ", fontsize=12)
xlims!(plt_prob, 0.0, duration)
ylims!(plt_prob, 0.0, 1.0)


# Save plots if flag is set
if save_plots
    figures_dir = joinpath(@__DIR__, "..", "figures")
    mkpath(figures_dir)  # Create directory if it doesn't exist
    
    # Save probability plot as PDF
    savefig(plt_prob, joinpath(figures_dir, "van_der_pol_2D_ablation_m_probability.pdf"))
    
    println("Plots saved to figures directory")
end
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
vp_deg = 5 # Volume polynomial degree (fixed for this ablation)
deg_incr = 0
Δt_max_range = [0.01, 0.05, 0.1, 0.2, 1.0] # Δt_max range for ablation study
λ = 10.0
save_plots = true
##################


X, fx_hat = generate_data(true_system, 2000; domain_std=0.5, noise_std=0.01)

U, fu_hat = x_data_to_u_data(X, fx_hat, dtf)

# System regression
learned_rmodel = constrained_system_regression(U, fu_hat, model_degrees, reverse=true, λ=λ)

target_region_u = Rx_to_Ru(dtf, target_region)

# Compute monte carlo eval (only need to compute once)
euler_prob_traj, timestamps, hoeffding_bounds, bernstein_bounds = euler_probability_traj(target_region_u, duration, forward_model=-learned_rmodel, n_samples=10000, n_timesteps=100)
function sampler()
    return [rand(Normal(0.0, 0.5)) for i in 1:2]
end
euler_prob_traj_true, timestamps_true, hoeffding_bounds_true, bernstein_bounds_true = euler_probability_traj(target_region, duration, forward_model=true_system, n_samples=10000, n_timesteps=100, sampler=sampler)

# Store taylor splines for each Δt_max
tamed_ts_list = []
geo_ts_list = []

# Compute flowpipe and taylor splines for each Δt_max
println("Computing flowpipes and taylor splines for Δt_max range: $Δt_max_range")
for Δt_max in Δt_max_range
    println("  Computing for Δt_max = $Δt_max...")
    # Regenerate flowpipe for this Δt_max
    flow_pipe = compute_bernstein_reach_sets(learned_rmodel, target_region_u, duration, expansion_degree=fp_deg, Δt_max=Δt_max, deg_incr=deg_incr)
    
    # Compute taylor splines for this flowpipe
    tamed_ts = create_tamed_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
    geo_ts = create_geometric_taylor_spline(flow_pipe, learned_rmodel, vp_deg)
    push!(tamed_ts_list, tamed_ts)
    push!(geo_ts_list, geo_ts)
end

## Plot the flowpipe
#plt_fp = plot(aspect_ratio = :equal)
#plt_fp = plot_flowpipe!(plt_fp, flow_pipe; color=:teal, alpha=0.0, n=10)
#plt_fp = plot_2D_region!(plt_fp, target_region_u; color=:red)

# Plot the probability functions
plt_prob = plot(legend=:topleft, tickfontsize=12, legendfontsize=12)  

# Plot MC learned and ground truth first
#plt_prob = plot(plt_prob, timestamps, euler_prob_traj, label="MC (learned)", color=:orange, linewidth=2)
#plt_prob = plot(plt_prob, timestamps_true, euler_prob_traj_true, label="MC (ground truth)", color=:purple, linewidth=2)

# Assign distinct colors to each Δt_max
# Using distinguishable colors: blue, red, green, orange, purple, brown, pink, gray
colors_by_dt_max = Dict(
    0.01 => :blue,
    0.05 => :red,
    0.1 => :green,
    0.2 => :orange,
    0.025 => :purple,
    0.15 => :brown,
    0.4 => :gray,
    1.0 => :pink
)

# Plot all tamed taylor splines (bound 1) - solid lines
for (Δt_max, tamed_ts) in zip(Δt_max_range, tamed_ts_list)
    global plt_prob
    color = colors_by_dt_max[Δt_max]
    plt_prob = plot_taylor_spline!(plt_prob, tamed_ts, duration, label="Bound 1 (Δt_max=$Δt_max)", color=color, linestyle=:solid)
end

# Plot all geo taylor splines (bound 2) - dashed lines
for (Δt_max, geo_ts) in zip(Δt_max_range, geo_ts_list)
    global plt_prob
    color = colors_by_dt_max[Δt_max]
    plt_prob = plot_taylor_spline!(plt_prob, geo_ts, duration, label="Bound 2 (Δt_max=$Δt_max)", color=color, linestyle=:dash)
end

## Concentration bounds
#plt_prob = plot!(plt_prob, timestamps, euler_prob_upper, linestyle=:dot, color=:red, label=nothing)
#plt_prob = plot!(plt_prob, timestamps_true, euler_prob_upper_true, linestyle=:dot, color=:purple, label=nothing)

xlabel!(plt_prob, "τ", fontsize=12)
xlims!(plt_prob, 0.0, duration)
ylims!(plt_prob, 0.0, 1.0)

#plot(plt_fp, plt_prob, layout=(1,3), size=(1200, 400))

# Save plots if flag is set
if save_plots
    figures_dir = joinpath(@__DIR__, "..", "figures")
    mkpath(figures_dir)  # Create directory if it doesn't exist
    
    ## Save flowpipe plot as PNG
    #savefig(plt_fp, joinpath(figures_dir, "van_der_pol_2D_flowpipe.pdf"))
    #savefig(plt_fp, joinpath(figures_dir, "van_der_pol_2D_flowpipe.png"))
    
    # Save probability plot as PDF
    savefig(plt_prob, joinpath(figures_dir, "van_der_pol_2D_ablation_dt_probability.pdf"))
    
    println("Plots saved to figures directory")
end
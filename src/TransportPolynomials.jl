module TransportPolynomials


using LazySets
using IterTools
using Distributions
using LinearAlgebra
using Plots
using Random
using DynamicPolynomials
using MultivariatePolynomials
using SumOfSquares
using JuMP
using MosekTools
using Symbolics
using BellBruno

include("DataStructures.jl")
include("Visualizaton.jl")
include("SystemRegression.jl")
include("VolumePolynomial.jl")
include("PolynomialBounds.jl")
include("Probability.jl")

# DataStructures
export SystemModel, SpatioTemporalPoly, TemporalPoly

# Visualization
export plot_data, plot_polynomial_surface, plot_2D_pdf, 
    plot_2D_erf_space_pdf, plot_2D_region, plot_2D_pdf, plot_2D_erf_space_vf,
    plot_vol_poly_density_vs_time, plot_euler_density_vs_time, plot_integ_poly_prob_vs_time,
    plot_euler_mc_prob_vs_time

# VolumePolynomial
export compute_coefficients, create_vol_poly, create_vol_poly_and_nxt_coeff, 
    create_integrator_poly, create_basic_sos_bound_poly, evaluate_integral, 
    density, euler_density, probability, mc_euler_probability

# Probability
export density, probability, evaluate_integral, propagate_sample, euler_density, mc_euler_probability

# PolynomialBounds
export sos_bound, sos_coeff_mag_bound

end
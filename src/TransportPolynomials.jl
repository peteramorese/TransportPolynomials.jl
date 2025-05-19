module TransportPolynomials


using DynamicPolynomials
using MultivariatePolynomials
using LazySets
using IterTools
using Distributions
using LinearAlgebra
using Plots

using Random

# DataStructures
export SystemModel, SpatioTemporalPoly

# Visualization
export plot_data, plot_polynomial_surface, plot_2D_pdf, 
    plot_2D_erf_space_pdf, plot_2D_pdf, plot_2D_erf_space_vf

# VolumePolynomial
export compute_coefficients, create_vol_poly, create_integrator_polynomial,
    evaluate_integral, density, euler_density, probability


include("DataStructures.jl")
include("Visualizaton.jl")
include("SystemRegression.jl")
include("VolumePolynomial.jl")

end
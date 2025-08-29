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
using ModelingToolkit
using IntervalArithmetic
using SpecialFunctions
using FFTW
#using Symbolics
#using BellBruno

import ReachabilityAnalysis as RA 

include("DataStructures.jl")
include("Visualizaton.jl")
include("SystemRegression.jl")
include("PolynomialBounds.jl")
include("VolumePolynomial.jl")
include("BoundedVolumePolynomial.jl")
include("Probability.jl")
include("PicardIteration.jl")
include("ReachableSets.jl")
include("TaylorSpline.jl")
include("BernsteinPolynomial.jl")

# DataStructures
export SystemModel, SpatioTemporalPoly, SpatioTemporalPolyVector, TemporalPoly

# Visualization
export plot_data, plot_polynomial_surface, plot_2D_pdf, 
    plot_2D_erf_space_pdf, plot_2D_region, plot_2D_region_in_3D, plot_2D_pdf, plot_2D_erf_space_vf,
    plot_vol_poly_density_vs_time, plot_euler_density_vs_time, plot_integ_poly_prob_vs_time,
    plot_euler_mc_prob_vs_time, plot_2D_reachable_sets

# VolumePolynomial
export compute_coefficients, create_vol_poly, create_vol_poly_and_nxt_coeff, 
    create_integrator_poly, create_basic_sos_bound_poly, create_basic_intarith_bound_poly, evaluate_integral, 
    density, euler_density, probability, mc_euler_probability

# BoundedVolumePolynomial
export TensorErrorVariables, create_tensor_error_variables

# Probability
export density, probability, evaluate_integral, propagate_sample, euler_density, mc_euler_probability

# PolynomialBounds
export BouldType, Upper, Lower, Magnitude
export sos_bound, dsos_bound, intarith_bound, coeff_sos_bound, coeff_intarith_bound

# PicardIteration
export picard_operator, picard_vol_poly

# ReachableSets
export compute_taylor_reach_sets, compute_final_hyperrectangle

# TaylorSpline
export TaylorSplineSegment, TaylorSpline, create_box_taylor_spline, create_continuous_taylor_spline 

# BernsteinPolynomial
export BernsteinPolynomial, degree, dimension, decasteljau, differentiate, integrate, product

end
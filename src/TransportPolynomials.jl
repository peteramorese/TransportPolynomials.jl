module TransportPolynomials


using StaticArrays
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
using LoopVectorization

import ReachabilityAnalysis as RA 

include("BernsteinPolynomial.jl")
include("DataStructures.jl")
include("BernsteinReachSets.jl")
include("DistributionTransform.jl")
include("SystemRegression.jl")
include("Systems.jl")
include("Euler.jl")
#include("PolynomialBounds.jl")
include("VolumePolynomial.jl")
#include("BoundedVolumePolynomial.jl")
include("Probability.jl")
#include("PicardIteration.jl")
include("ReachableSets.jl")
include("TaylorSpline.jl")
include("Visualizaton.jl")

# DataStructures
export SystemModel, dimension, SpatioTemporalPoly

# BernsteinPolynomial
export BernsteinPolynomial, deg, dimension, decasteljau, log_eval, differentiate, integrate, product, increase_degree, add, add!, affine_transform, upper_bound, lower_bound, to_mv_polynomial

# BernsteinReachSets
export create_bernstein_expansion, create_bernstein_field_expansion, compute_bernstein_reach_sets, reposition, get_final_region

# Visualization
export plot_2D_region!, plot_taylor_spline!, plot_flowpipe!

# DistributionTransform
export DistributionTransform, dimension, x_to_u, u_to_x, Rx_to_Ru, Ru_to_Rx, to_state_space_model, to_u_space_model

# SystemRegression
export generate_data, x_data_to_u_data, constrained_poly_regression, constrained_system_regression

# Systems
export van_der_pol, cartpole

# Euler
export euler_density, euler_probability, euler_probability_traj

# VolumePolynomial
export compute_coefficients, create_vol_poly, create_vol_poly_and_nxt_coeff 

# Probability
export density, probability, propagate_sample, euler_density, mc_euler_probability

# PolynomialBounds
export BoundType, Upper, Lower, Magnitude
export sos_bound, dsos_bound, intarith_bound, coeff_sos_bound, coeff_intarith_bound

# PicardIteration
export picard_operator, picard_vol_poly

# ReachableSets
export compute_taylor_reach_sets, to_mv_polynomial_system, compute_hyperrectangle

# TaylorSpline
export TaylorSplineSegment, TaylorSpline, create_box_taylor_spline, create_tamed_taylor_spline 


end
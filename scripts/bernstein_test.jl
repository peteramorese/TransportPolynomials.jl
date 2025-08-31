using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using LazySets

#coeffs = Array{Float64, 3}(undef, 3, 3, 2)
#coeffs[:, :, 1] = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
#coeffs[:, :, 2] = [2.0 4.0 6.0; 3.0 5.0 7.0; 4.0 8.0 10.0]

#coeffs = Array{Float64, 3}(undef, 2, 2, 2)
#coeffs = [1.9 2.9; 4.9 5.9;; 2.1 4.1; 3.1 5.1]
#coeffs[:, :, 1] = [1.9 2.9; 4.9 5.9]
#coeffs[:, :, 2] = [2.1 4.1; 3.1 5.1]
coeffs = reshape([1.9, 2.1, 4.9, 3.1, 2.9, 4.1, 5.9, 5.1], 2, 2, 2)
coeffs_p = reshape([1.0, 2.0, 4.0, 3.0, 7.0, 4.0, 2.0, 4.0, 5.0, 5.0, 8.0, 8.0, 3.0, 6.0, 6.0, 7.0, 9.0, 10.0], 2, 3, 3)
#coeffs_q = reshape([1.5, 2.5, 4.5, 3.5, 7.5, 4.5, 2.5, 4.5, 5.5, 5.5, 8.5, 8.5, 3.5, 6.5, 6.5, 7.5, 9.5, 10.5], 2, 3, 3)


p = BernsteinPolynomial{Float64, 3}(coeffs_p)

p_mono = to_mv_polynomial(p)

#sub_region = Hyperrectangle(low=[.2, .3, .2], high=[0.4, 0.4, 0.4])
#p_tf = affine_transform(p, sub_region)

#println("Degree: " , deg(p))
#m = (5, 6, 7)
#p_incr = increase_degree(p, m)
#println("pincr coeffs: \n", p_incr.coeffs)

#q = BernsteinPolynomial{Float64, 3}(coeffs_q)
#prod = product(p, q)
#x = [0.5 0.5 0.3; 0.25 0.75 0.8]
x = [0.3, 0.35, 0.3]
#x_tf = [0.5, 0.5, 0.5]

println("De casteljau p: ", decasteljau(p, x))
println("MV poly p: ", p_mono(x))
#println("De casteljau ptf: ", decasteljau(p_tf, x_tf))
#println("De casteljau p_incr: ", decasteljau(p_incr, x))
#println("prod De casteljau: ", decasteljau(prod, x))
#println("integral: ", integrate(p, Hyperrectangle(low=[0.1, 0.2, 0.3], high=[0.5, 0.6, 0.7])))
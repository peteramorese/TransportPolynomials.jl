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
#coeffs = reshape([1.9, 2.1, 4.9, 3.1, 2.9, 4.1, 5.9, 5.1], 2, 2, 2)
coeffs = reshape([1.0, 2.0, 4.0, 3.0, 7.0, 4.0, 2.0, 4.0, 5.0, 5.0, 8.0, 8.0, 3.0, 6.0, 6.0, 7.0, 9.0, 10.0], 2, 3, 3)



p = BernsteinPolynomial{Float64, 3}(coeffs)
x = [0.5 0.5 0.3; 0.25 0.75 0.8]
#x = [0.5, 0.5, 0.3]
println("De casteljau: ", decasteljau(p, x))
println("integral: ", integrate(p, Hyperrectangle(low=[0.1, 0.2, 0.3], high=[0.5, 0.6, 0.7])))
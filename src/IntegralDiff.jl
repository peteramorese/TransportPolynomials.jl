function integral_diff(p::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}
    pos_part_integral = integrate(pos_part(p), region)

end
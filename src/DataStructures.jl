struct SystemModel
    x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}
    f::Vector{<:MultivariatePolynomials.AbstractPolynomialLike}
end

# Evaluate the polynomial vector field at the given point
function (model::SystemModel)(x::Vector{Float64})
    return Vector{Float64}([subs(model.f[i], Tuple(model.x_vars) => Tuple(x)) for i in 1:length(model.f)])
end

function dimension(model::SystemModel)
    return length(model.x_vars)
end

struct SpatioTemporalPoly
    x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}
    t_var::Variable
    p::AbstractPolynomialLike
end

# Evaluate the spatio-temporal polynomial at the givem point and time
function (stp::SpatioTemporalPoly)(x::Vector{Float64}, t::Float64)
    subst = Dict(stp.x_vars[i] => x[i] for i in 1:length(stp.x_vars))
    merge!(subst, Dict(stp.t_var => t))
    return convert(Float64, subs(stp.p, subst...))
end

struct TemporalPoly
    t_var::Variable
    p::AbstractPolynomialLike
end

function (tp::TemporalPoly)(t::Float64)
    return convert(Float64, subs(tp.p, tp.t_var => t))
end
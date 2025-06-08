struct SystemModel
    x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}
    f::Vector{<:MultivariatePolynomials.AbstractPolynomialLike}
end

# Evaluate the polynomial vector field at the given point
function (model::SystemModel)(x::Vector{Float64})
    return Vector{Float64}([subs(model.f[i], Tuple(model.x_vars) => Tuple(x)) for i in 1:length(model.f)])
end

function dimension(model::SystemModel)::Int64
    
    return length(model.f)
end

struct SpatioTemporalPoly
    x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}
    t_var::MultivariatePolynomials.AbstractVariable
    p::AbstractPolynomialLike
end

# Evaluate the spatio-temporal polynomial at the givem point and time
function (stp::SpatioTemporalPoly)(x::Vector{Float64}, t::Float64)
    subst = Dict(stp.x_vars[i] => x[i] for i in 1:length(stp.x_vars))
    merge!(subst, Dict(stp.t_var => t))
    return convert(Float64, subs(stp.p, subst...))
end

mutable struct SpatioTemporalPolyVector
    x_vars::Vector{<:MultivariatePolynomials.AbstractVariable}
    t_var::MultivariatePolynomials.AbstractVariable
    p::Vector{<:AbstractPolynomialLike}
end

# Evaluate the spatio-temporal polynomial vector at the givem point and time
function (stpv::SpatioTemporalPolyVector)(x::Vector{Float64}, t::Float64)
    subst = Dict(stpv.x_vars[i] => x[i] for i in 1:length(stpv.x_vars))
    merge!(subst, Dict(stpv.t_var => t))
    return [convert(Float64, subs(stpv.p[i], subst...)) for i in 1:length(stpv.x_vars)]
end

function dimension(stpv::SpatioTemporalPolyVector)
    return length(stpv.x_vars)
end

struct TemporalPoly
    t_var::MultivariatePolynomials.AbstractVariable
    p::AbstractPolynomialLike
end

function (tp::TemporalPoly)(t::Float64)
    return convert(Float64, subs(tp.p, tp.t_var => t))
end

function differentiate(tp::TemporalPoly)
    # Differentiate the temporal polynomial with respect to the time variable
    return TemporalPoly(tp.t_var, differentiate(tp.p, tp.t_var))
end

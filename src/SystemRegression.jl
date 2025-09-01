
# Generate synthetic data
function generate_data(true_system::SystemModel, n; domain_std=1.0, noise_std=0.1, seed=0)
    Random.seed!(seed)
    D = dimension(true_system)
    Sx = randn(n, D) .* domain_std
    fx_clean = zeros(n, D)
    fx_clean = true_system(X)
    #for i in 1:n
    #    Y_clean[i, :] = true_system(X[i, :])
    #end
    noise = randn(n, D) .* noise_std
    fx_noisy = fx_clean + noise
    return Sx, fx_noisy
end

function poly_regression(polyvars, S::Matrix{Float64}, f_hat_component::Vector{Float64}; degrees::Vector{Int})
    n, D = size(S)
    @assert length(f_hat) == n "Number of data in S and f_hat must match"

    shape = tuple(degrees .+ 1)

    n_basis_functions = prod(shape)

    A = zeros(n, n_basis_functions)

    log_S = log.(S)
    log_1mS = log.(1 .- S)

    wrap_i = 0
    for I in CartesianIndices(shape)
        log_basis = zeros(n)
        for j in I
            idx = j - 1
            deg = degrees[j]

            log_binom = lgamma(deg + 1) - lgamma(idx + 1) - lgamma(deg - idx + 1)
            log_basis += log_binom .+ idx * log_S[:, j] .+ (deg - idx) * log_1mS[:, j]
        end
        A[:, i] = exp.(log_basis)
        i += 1
    end
        
    # Least squares
    coeffs = A \ f_hat

    # Form the polynomial from monomials and fitted coefficients
    coeffs = reshape(coeffs, shape)
    return BernsteinPolynomial{Float64, D}(coeffs)
end

function system_regression(S, Y_u, degree)
    @assert size(X_u) == size(Y_u)

    d = size(X_u, 2)
    @polyvar x[1:d]

    model = Vector{Polynomial}()
    
    bc_scaling = 1 ./(X_u .* (1 .- X_u))
    Y_u_unconst = Y_u .* bc_scaling

    for i in 1:n
        y = Y_u_unconst[:, i]
        p = poly_regression(x, X_u, y, deg=degree) 
        p_bc = x[i] * (1 - x[i]) * p 
        push!(model, p_bc)
    end
    return model
end


# Example usage
d = 2
n = 1000
degree = 3

example_function(x) = sin.(4 * x) .* 1 ./ (1 .+ (.5*x).^2)
X, Y = generate_data(example_function, d, n, noise_std=0.01)

#visualize_data(X, Y)

# Transform to unit box
X_u = erf_space_transform.(X)
Y_u = zeros(size(Y))
for i in 1:size(X, 1)
    J = erf_space_transform_jacobian(X[i, :])
    Y_u[i, :] = J * Y[i, :]
end

#visualize_data(X_u, Y_u, "Erf space transformed data")

# Fit constrained model
#models = fit_constrained_polynomial(X_u, Y_u, degree)












#@polyvar x[1:2]
#
#X = randn(n, 2)
#y = .4 * X[:,1].^3 + 2 * X[:,2].^2 .+ 5 .+ 0.1 * randn(n)
#
## Perform polynomial regression
#p = poly_regression(x[1:2], X, y, deg=3)
#
#p_true = .4 * x[1]^3 + 2 * x[2]^2 .+ 5
#
#println("Fitted polynomial:")
#println(p)
#
#p1 = scatter(X[:, 1], X[:, 2], y, markersize=4, title="f1(x1, x2)", xlabel="x1", ylabel="x2", zlabel="f1")
#p2 = plot_polynomial_surface(p, x[1], x[2], (-3, 3), (-3, 3), title="Polynomial Surface")
#p3 = plot_polynomial_surface(p_true, x[1], x[2], (-3, 3), (-3, 3), title="True Polynomial Surface")
#display(plot(p1, p2, p3, layout=(1, 3), size=(800, 400)))


# Generate synthetic data
function generate_data(true_system::SystemModel, n; domain_std=1.0, noise_std=0.1, seed=0)
    Random.seed!(seed)
    D = dimension(true_system)
    X = randn(n, D) .* domain_std
    fx_clean = zeros(n, D)
    fx_clean = true_system(X)
    #for i in 1:n
    #    Y_clean[i, :] = true_system(X[i, :])
    #end
    noise = randn(n, D) .* noise_std
    fx_hat = fx_clean + noise
    return X, fx_hat
end

function x_data_to_u_data(X::Matrix{Float64}, fx_hat::Matrix{Float64}, dtf::DistributionTransform)
    n, D = size(X)
    @assert size(fx_hat, 1) == n
    @assert size(X, 2) == size(fx_hat, 2)

    U = x_to_u(dtf, X)
    #print(size(diag(pdf.(dtf.dist, X))))
    fu_hat = similar(fx_hat)
    for i in 1:D
        fu_hat[:, i] = pdf.(Ref(dtf.dist[i]), X[:, i]) .* fx_hat[:, i]
    end
    return U, fu_hat
end

function constrained_poly_regression(constrained_dim::Int, U::Matrix{Float64}, f_hat_component::Vector{Float64}; degrees::Vector{Int})
    n, D = size(U)
    @assert length(f_hat_component) == n "Number of data in U and f_hat must match"

    shape = tuple((degrees .+ 1)...)

    # We don't regress the i=0 or i=deg coefficients since they are constrained to be 0
    free_coeffs_shape = degrees .+ 1
    free_coeffs_shape[constrained_dim] -= 2
    free_coeffs_shape = tuple(free_coeffs_shape...)

    n_basis_functions = prod(free_coeffs_shape)

    A = zeros(n, n_basis_functions)

    log_S = log.(U)
    log_1mS = log.(1 .- U)

    i = 1
    for I in CartesianIndices(free_coeffs_shape)
        log_basis = zeros(n)
        for j in 1:D
            if j != constrained_dim
                idx = I[j] - 1
            else
                idx = I[j] + 1
            end
            deg = degrees[j]

            log_binom = lgamma(deg + 1) - lgamma(idx + 1) - lgamma(deg - idx + 1)
            log_basis += log_binom .+ idx * log_S[:, j] .+ (deg - idx) * log_1mS[:, j]
        end
        A[:, i] = exp.(log_basis)
        i += 1
    end
        
    # Least squares
    free_coeffs = A \ f_hat_component
    free_coeffs = reshape(free_coeffs, free_coeffs_shape)

    indices = [j == constrained_dim ? (2:degrees[j]) : Colon() for j in 1:D]

    coeffs = zeros(Float64, shape...)
    coeffs[indices...] = free_coeffs
    return BernsteinPolynomial{Float64, D}(coeffs)
end

function constrained_system_regression(U::Matrix{Float64}, f_hat::Matrix{Float64}, degrees::Vector{Int}; reverse::Bool=false)
    n, D = size(U)
    @assert size(f_hat, 1) == n "Number of data in U and f_hat must match"
    @assert size(f_hat, 2) == D "Dimension of f_hat must match dimension of U"
    @assert length(degrees) == D "Length of degrees must match dimension of U"

    if reverse
        f_hat *= -1.0
    end

    f_polys = Vector{BernsteinPolynomial{Float64, D}}(undef, D)
    for i in 1:D
        f_polys[i] = constrained_poly_regression(i, U, f_hat[:, i], degrees=degrees)
    end
    return SystemModel(f_polys)
end


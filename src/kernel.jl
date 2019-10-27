function rbf(γ::AbstractArray, X::Array)::Array
    n_samples = size(X)[2]
    K = Array{eltype(γ)}(undef, n_samples, n_samples)
    for i in 1:n_samples
        K[i,i] = 1.0
        for j in (i+1):n_samples
            K[i,j] = K[j,i] = exp(sum(-γ.*((X[:,i]-X[:,j]).^2)))
        end
    end
    return K
end

function rbf(γ::AbstractArray, X::Array, x::Array)::Array
    n_samples = size(X)[2]
    n_test = size(x)[2]
    K = Array{eltype(γ)}(undef, n_samples, n_test)
    for i in 1:n_samples
        for j in 1:n_test
            K[i,j] = exp(sum(-γ.*((X[:,i]-x[:,j]).^2)))
        end
    end
    return K
end

function ∇ₓrbf(γ::AbstractArray, X::Array, x::Array)::Array
    n_features = size(X)[1]
    n_samples = size(X)[2]
    n_test = size(x)[2]
    ∇ₓK = Array{eltype(γ)}(undef, n_features, n_samples, n_test)
    for i in 1:n_samples
        for j in 1:n_test
            ∇ₓK[:, i, j] = 2γ.*(X[:, i]-x[:, j]).*exp(sum(-γ.*((X[:, i]-x[:, j]).^2)))
        end
    end
    return ∇ₓK
end

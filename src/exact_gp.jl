struct GaussProcess{T, D, F0, F1, F2, F3}
    γ::T
    β::D
    μ::F0
    ∇ₓμ::F1
    Σ::F2
    ∇ₓΣ::F3
end

# function GaussProcess(γ::Array{T}, β::T, μ, ∇ₓμ, Σ, ∇ₓΣ) where T
#     return GaussProcess(param(γ), β, μ, ∇ₓμ, Σ, ∇ₓΣ)
# end

function GaussProcess(in_dim::Integer, β::Real, μ, ∇ₓμ, Σ, ∇ₓΣ)
    γ = param(rand(in_dim))
    return GaussProcess(γ, β, μ, ∇ₓμ, Σ, ∇ₓΣ)
end

Flux.@treelike(GaussProcess)

function negloglik(GP::GaussProcess, X̄::Matrix, ȳ::Vector)
    K = GP.Σ(relu.(GP.γ), X̄, X̄) + (GP.β^2)*I   # K is a Array{TrackedReal, 2}
    μ = GP.μ(X̄)         # mu is a TrackedArray{T, 1}
    likelihood = MvNormal(μ, K)
    nll = -1*logpdf(likelihood, ȳ)
    return nll
end

function predict(GP::GaussProcess, X̄::Matrix, ȳ::Vector, x::Matrix)
    K = GP.Σ(Flux.data(GP.γ), X̄, X̄) + (GP.β^2)*I
    k = GP.Σ(Flux.data(GP.γ), X̄, x)
    reduced_ȳ = ȳ - Flux.data(GP.μ(X̄))

    Cho_K = cholesky(K)
    μ = Flux.data(GP.μ(x)) + k'*(Cho_K\reduced_ȳ)
    Σp = GP.Σ(Flux.data(GP.γ), x, x) - k'*(Cho_K\k)

    σ = sqrt.(diag(Σp))
    return μ, σ
end

function ∇ₓpredict(GP::GaussProcess, X̄::Matrix, ȳ::Vector, x::Matrix)
    n_features, n_test = size(x)
    K = GP.Σ(relu.(GP.γ), X̄, X̄) + (GP.β^2)*I
    ∇ₓk = GP.∇ₓΣ(relu.(GP.γ), X̄, x)
    ∇ₓμ₀ = GP.∇ₓμ(x)      # Array{TrackedReal, 1}
    reduced_ȳ = ȳ - GP.μ(X̄)

    Cho_K = cholesky(K)
    α = Cho_K\reduced_ȳ
    ∇ₓμ₁ = Array{eltype(GP.γ)}(undef, n_features, n_test)
    for i in 1:n_test
        ∇ₓμ₁[:, i] = ∇ₓk[:, :, i]*α
    end
    return ∇ₓμ₀ + ∇ₓμ₁
end

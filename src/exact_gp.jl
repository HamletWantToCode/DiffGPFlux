struct GaussProcess{T, D, F1, F2}
    γ::T
    β::D
    Σ::F1
    ∇ₓΣ::F2
end

function GaussProcess(in_dim::Integer, β::Real, Σ, ∇ₓΣ)
    γ = param(rand(in_dim))
    return GaussProcess(γ, β, Σ, ∇ₓΣ)
end

function GaussProcess(γ::Array{T}, β::T, Σ, ∇ₓΣ) where T
    return GaussProcess(param(γ), β, Σ, ∇ₓΣ)
end

Flux.@treelike(GaussProcess)


function negloglik(GP::GaussProcess, X̄::Array, ȳ::Vector)
    n_samples = size(X̄)[2]
    K = GP.Σ(relu.(GP.γ), X̄) + (GP.β^2)*I
    likelihood = MvNormal(K)
    nll = -1*logpdf(likelihood, ȳ)
    return nll
end

function predict(GP::GaussProcess, X̄::Array, ȳ::Vector, x::Array)
    n_samples = size(X̄)[2]
    K = GP.Σ(Flux.data(GP.γ), X̄) + (GP.β^2)*I
    k = GP.Σ(Flux.data(GP.γ), X̄, x)

    Cho_K = cholesky(K)
    μ = k'*(Cho_K\ȳ)
    Σp = GP.Σ(Flux.data(GP.γ), x) - k'*(Cho_K\k)

    σ = sqrt.(diag(Σp))
    return μ, σ
end

function ∇ₓpredict(GP::GaussProcess, X̄::Array, ȳ::Vector, x::AbstractArray)
    n_features, n_test = size(x)
    K = GP.Σ(relu.(GP.γ), X̄) + (GP.β^2)*I
    ∇ₓk = GP.∇ₓΣ(relu.(GP.γ), X̄, x)

    Cho_K = cholesky(K)
    α = Cho_K\ȳ
    ∇ₓμ = Array{eltype(GP.γ)}(undef, n_features, n_test)
    for i in 1:n_test
        ∇ₓμ[:, i] = ∇ₓk[:, :, i]*α
    end
    return ∇ₓμ
end

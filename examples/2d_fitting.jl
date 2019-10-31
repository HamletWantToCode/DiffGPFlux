using DiffGPFlux
using Random: randperm
using Plots

f(x) = [1.0 0.0] * (sin.(x) + cos.([0.0 1.0; 1.0 0.0]*x))
∂f(x) = [1.0 0.0; 0.0 0.0]*cos.(x) - [0.0 0.0; 0.0 1.0]*sin.(x)

X = Array(range(-2, 2, length=50))
Y = Array(range(-1, 3, length=50))
XX = repeat(X, 1, 50)
YY = repeat(Y', 50, 1)

XY = vcat(reshape(XX, 1, 2500), reshape(YY, 1, 2500))
Z = dropdims(f(XY), dims=1) + 0.1*randn(2500)
∂Z = ∂f(XY) + 0.1.*randn(2, 2500)
Index = randperm(size(XY, 2))
train_index = Index[1:1500]
test_index = Index[1501:end]
XY_train, Z_train, ∂Z_train = XY[:, train_index], Z[train_index], ∂Z[train_index]
XY_test, Z_test, ∂Z_test = XY[:, test_index], Z[test_index], ∂Z[test_index]

γ₀ = seed_duals([0.3, 1.0], Float64, 2)
β = 0.1
GP = GaussProcess(γ₀, β, rbf, ∇ₓrbf)
θ = [GP.γ]
indexes = [[1, 2]]

opt = ADAM(0.01)

plt = plot(1, marker=2)
@gif for i in 1:100
    nll = negloglik(GP, XY_train, Z_train)
    @show(nll.value)
    push!(plt, nll.value)
    θ̄_array = [g for g in nll.partials]
    θ̄ = grads(θ̄_array, indexes)
    update!(opt, θ, θ̄, 2)
    end every 5

GP.γ

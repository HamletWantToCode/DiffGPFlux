using DiffGPFlux
using Random: randperm
using Plots

f(x) = [1.0 0.0] * (sin.(x) + cos.([0.0 1.0; 1.0 0.0]*x))
∂f(x) = [1.0 0.0; 0.0 0.0]*cos.(x) - [0.0 0.0; 0.0 1.0]*sin.(x)

X = Array(range(-2, 2, length=30))
Y = Array(range(-1, 3, length=30))
XX = repeat(X, 1, 30)
YY = repeat(Y', 30, 1)

XY = vcat(reshape(XX, 1, 900), reshape(YY, 1, 900))
Z = dropdims(f(XY), dims=1) + 0.1*randn(900)
∂Z = ∂f(XY) + 0.1.*randn(2, 900)
Index = randperm(size(XY, 2))
train_index = Index[1:500]
test_index = Index[501:end]
XY_train, Z_train, ∂Z_train = XY[:, train_index], Z[train_index], ∂Z[train_index]
XY_test, Z_test, ∂Z_test = XY[:, test_index], Z[test_index], ∂Z[test_index]

γ_dim = 2
β = 0.1
μ(x) = zeros(size(x, 2))
∇ₓμ(x) = zeros(size(x))
GP = GaussProcess(γ_dim, β, μ, ∇ₓμ, rbf, ∇ₓrbf)
ps = Flux.params(GP)

opt = ADAM(0.01)

plt = plot(1, marker=2)
@gif for i in 1:100
    nll = negloglik(GP, XY_train, Z_train)
    display(Tracker.data(nll))
    push!(plt, Tracker.data(nll))
    Tracker.back!(nll)
    for p in ps
        Tracker.update!(opt, p, p.grad)
    end
    end every 5

GP.γ

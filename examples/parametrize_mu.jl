using DiffGPFlux
using Measurements
using Random
using Plots


# This is a function with a linear trend, we want to see
# whether NN parametrized GP mean function is able to find
# the trend
X = reshape(Array(range(0.0, 10.0, length=100)), 1, 100)
y = dropdims(0.5X.*sin.(X) + 2X, dims=1) + 0.1randn(100)
Index = randperm(100)
train_index = Index[1:30]
test_index = Index[31:end]
X_train, y_train = X[:, train_index], y[train_index]
X_test, y_test = X[:, test_index], y[test_index]

# build model
γ = [1.0]
β = 0.1
lin = Dense(1, 1) |> f64
μ = Chain(lin, x -> dropdims(x, dims=1))
∇ₓμ = nothing
GP = GaussProcess(γ, β, μ, ∇ₓμ, rbf, ∇ₓrbf)
ps = Flux.params(GP)

opt = ADAM(0.01)

plt = plot(1, marker=2)
@gif for i in 1:200
    nll = negloglik(GP, X_train, y_train)
    push!(plt, Tracker.data(nll))
    Tracker.back!(nll)
    for p in ps
        Tracker.update!(opt, p, p.grad)
    end
    end every 5

display(GP.γ)
display(GP.μ.layers[1].W)
display(GP.μ.layers[1].b)

y_predict, σ_predict = predict(GP, X_train, y_train, X)
plot(X_train', y_train, seriestype=:scatter)
plot!(X', y_predict.±σ_predict, seriestype=:scatter)
plot!(X', y, color=:red, lw=3)

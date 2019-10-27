using DiffGPFlux
using Measurements
using Plots
using Random: randperm

# build training and testing data,
# for X array along the 1st dimension is the feature,
# and along the 2nd dimension is the data samples,
# y array is required to be a Vector.
X = reshape(Array(range(0,2π,length=100)), 1, 100)
y = dropdims(sin.(X) + 0.01 .* randn(1, 100), dims=1)

index = randperm(100)
train_index = index[1:20]
test_index = index[21:end]
X_train, y_train = X[:, train_index], y[train_index]
X_test, y_test = X[:, test_index], y[test_index]

# build the Gauss process model
γ = [0.8]
β = 0.1
GP = GaussProcess(γ, β, rbf, ∇ₓrbf)
ps = Flux.params(GP)

opt = ADAM(0.008)

plt = plot(1, marker=2)
@gif for i in 1:200
    nll = negloglik(GP, X_train, y_train)
    push!(plt, Tracker.data(nll))
    Tracker.back!(nll)
    for p in ps
        Tracker.update!(opt, p, p.grad)
    end
    end every 5

display(GP.γ) # 0.15405557038969345

y_predict, σ_predict = predict(GP, X_train, y_train, X)
plot(X_train', y_train, seriestype=:scatter)
plot!(X', y_predict.±σ_predict, seriestype=:scatter)
plot!(X', y, color=:red, lw=3)

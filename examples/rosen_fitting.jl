using DiffGPFlux
using Random: randperm
using Plots


rosen(x) = (1 .+ [-1.0 0.0]*x).^2 + 100([0.0 1.0]*x+[-1.0 0.0]*x.^2).^2
rosen_der(x) = 2(1 .+ [-1.0 0.0]*x).*[-1.0, 0.0] + 200([0.0 1.0]*x+[-1.0 0.0]*x.^2).*([0.0, 1.0].+[-2.0, 0.0].*x)

X = Array(range(-2.0, 2.0, 50))
Y = Array(range(-1.0, 3.0, 50))
XX = repeat(X, 1, 50)
YY = repeat(Y', 50, 1)

XY = vcat(reshape(XX, 1, :), reshape(YY, 1, :))
Z = rosen(XY)
∂Z = rosen_der(XY)
Index = randperm(size(XY, 2))
train_index = Index[1:1000]
test_index = Index[1001:end]
XY_train, Z_train, ∂Z_train = XY[:, train_index], Z[train_index], ∂Z[train_index]
XY_test, Z_test, ∂Z_test = XY[:, test_index], Z[test_index], ∂Z[test_index]

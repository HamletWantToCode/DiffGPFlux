using DiffGPFlux
using Random
using PyPlot


function meshgrid(X, Y)
    len_X = length(X)
    len_Y = length(Y)
    XX = repeat(X', len_Y)
    YY = repeat(Y, 1, len_X)
    return XX, YY
end

function data_gen(f, fp, xrange, yrange)
    X = Array(range(xrange[1], xrange[2], length=50))
    Y = Array(range(yrange[1], yrange[2], length=50))
    XX, YY = meshgrid(X, Y)
    XY = vcat(reshape(XX, 1, 2500), reshape(YY, 1, 2500))
    _xy = zip(reshape(XX, 2500), reshape(YY, 2500))
    _XY = [[x,y] for (x,y) in _xy]
    Z = f.(_XY)
    _∂Z = fp.(_XY)
    ∂Z = hcat(_∂Z...)
    return X, Y, XY, Z, ∂Z
end

function GP_der_field(GP, X, y, x)
    K̄, k̄, K, μ₀, μ̄, σ̄ = pre_alloc(GP.γ, X, y, x)
    predict!(GP, X, y, x, K̄, k̄, K, μ̄, σ̄)
    return reshape(μ̄, size(x))
end

rosen(x) = (1-x[1])^2+100*(x[2]-x[1]^2)^2
rosen_der(x) = [-2(1-x[1])-400*x[1]*(x[2]-x[1]^2), 200*(x[2]-x[1]^2)]
xrange = [-2.0, 2.0]
yrange = [-1.0, 3.0]
X, Y, XY, Z, ∂Z = data_gen(rosen, rosen_der, xrange, yrange)
index = randperm(2500)
train_index = index[1:100]
train_XY, train_∂Z = XY[:, train_index], ∂Z[:, train_index]

plot_XY = XY

γ = [1.0, 1.0]
β = 0.1
μ = x -> zero(x)
GP = GaussProcess(γ, β, μ, identity_decomposable_kernel!)
y = vec(train_∂Z)
plot_∂Z = GP_der_field(GP, train_XY, y, XY)

xx = reshape(plot_XY[1, :], 50, 50)
yy = reshape(plot_XY[2, :], 50, 50)
gp_∂Zx = reshape(plot_∂Z[1, :], 50, 50)
gp_∂Zy = reshape(plot_∂Z[2, :], 50, 50)
gp_norm_∂Z = sqrt.(gp_∂Zx.^2 .+ gp_∂Zy.^2)
gp_lw = gp_norm_∂Z./maximum(gp_norm_∂Z)
streamplot(xx, yy, gp_∂Zx, gp_∂Zy, color="k", linewidth=gp_lw)
streamplot(xx, yy, ∂Zx, ∂Zy, color="r", linewidth=lw)
plt.savefig("test_stream.png")

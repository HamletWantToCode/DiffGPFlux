using DiffGPFlux
using Test

γ = rand(3)
X = rand(3, 10)

@testset "DiffGPFlux.jl" begin
    @test (K=rbf(γ, X, X); isposdef(K))   # verify postive definite
    @test_broken (K=linear(γ, X, X); issymmetric(K))   # numerical error will affect linear kernel !!!
    @test (K=rbf(γ, X, X); diag(K)==ones(size(K, 1)))  # for RBF kernel, diagonal elements should all be 1
    @test (K=rbf(γ, X, X); eltype(K)==eltype(γ))    # type of K should be the same as type of γ
end

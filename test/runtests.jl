using DiffGPFlux
using Test

X = rand(3, 10)
y = rand(10)
x = rand(3, 2)

function type_size_test(γ)
    K = rbf(γ, X)
    @assert size(K) == (10, 10)
    @assert typeof(K) == Array{eltype(γ), 2}
    k = rbf(γ, X, x)
    @assert size(k) == (10, 2)
    @assert typeof(k) == Array{eltype(γ), 2}
    return true
end

@testset "DiffGPFlux.jl" begin
    @test (γ=rand(1); type_size_test(γ))
    @test (γ=rand(3); type_size_test(γ))
    @test (γ=TrackedArray(rand(3)); type_size_test(γ))
end

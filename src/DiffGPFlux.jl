module DiffGPFlux

    using Reexport
    @reexport using Flux
    @reexport using Distributions
    @reexport using LinearAlgebra
    @reexport using Random

    include("exact_gp.jl")
    include("kernel.jl")

    export GaussProcess, negloglik, predict, ∇ₓpredict
    export rbf, ∇ₓrbf

end # module

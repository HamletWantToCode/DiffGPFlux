module DiffGPFlux

    using Reexport
    @reexport using ForwardDiff
    @reexport using Distributions
    @reexport using LinearAlgebra

    include("exact_gp.jl")
    include("kernel.jl")
    include("utils.jl")
    include("optimizers.jl")

    export GaussProcess, negloglik, predict, ∇ₓpredict
    export rbf, ∇ₓrbf
    export seed_duals, update!, grads
    export Descent, Momentum, Nesterov, RMSProp, ADAM, ADAGrad

end # module

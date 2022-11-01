using Flux

struct DensePosChain{F, M <: AbstractMatrix, B}
    positions::Vector{M},
    weights::Vector{M},
    biases::Vector{B},
    activations::Vector{F}
    # "ws" is weights, "as" is activations
    function DensePosChain(ws::Vector{M}, bias = true, as::Vector{F} = fill(identity, length(ws))) where {M <: AbstractMatrix, F}
        b = [create_bias(w, bias, size(w,1)) for w in ws]
        new{F,M,typeof(b)}(ws, b, as)
    end
end
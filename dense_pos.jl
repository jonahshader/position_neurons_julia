using Flux

struct DensePosChain{F <: AbstractVector, M <: AbstractVector{AbstractMatrix}, B <: AbstractVector}
    positions::M,
    weights::M,
    biases::B
    activations::F
end

# "ws" is weights, "as" is activations
function DensePosChain(ws::Vector{M}, bias = true, as::Vector{F} = fill(identity, length(ws))) where {M <: AbstractMatrix, F}
    b = [create_bias(w, bias, size(w,1)) for w in ws]
    new{F,M,typeof(b)}(ws, b, as)
end
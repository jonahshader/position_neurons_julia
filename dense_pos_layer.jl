using Flux
include("funs.jl")

struct DensePosLayer
    positions::AbstractMatrix
    previous_positions::AbstractMatrix
    weights::AbstractMatrix
    bias::AbstractVector
    activation::Function
    dist_fun::Function
end

function DensePosLayer(prev_pos::AbstractMatrix, output_size::Int; pos_dims = 2, activation = identity, dist_fun = bellcurve, mean::Union{AbstractVector, Nothing} = nothing, std::Union{AbstractVector, Nothing} = nothing)
    weights = Flux.glorot_uniform(output_size, size(prev_pos)[1])
    bias = zeros(Float32, output_size)
    positions = randn(Float32, output_size, pos_dims)
    if !isnothing(std)
        positions .*= std'
    end
    if !isnothing(mean)
        positions .+= mean'
    end
    weights ./= Float32(correct_scale(positions, prev_pos, dist_fun))
    
    return DensePosLayer(
        positions,
        prev_pos,
        weights,
        bias,
        activation,
        dist_fun
    )
end

Flux.@functor DensePosLayer
Flux.trainable(a::DensePosLayer) = (a.positions, a.weights, a.bias)

function (a::DensePosLayer)(x)
    activation = NNlib.fast_act(a.activation)
    # println(typeof(a.positions))
    # println(typeof(a.previous_positions))
    return activation.((a.weights .* a.dist_fun.(mm_dist2(a.positions, a.previous_positions))) * x .+ a.bias)
    # return activation.((a.weights) * x .+ a.bias)
end

function correct_scale(pos, previous_pos, dist_fun)
    sum(dist_fun.(mm_dist2(pos, previous_pos))) / (size(pos)[1] * size(previous_pos)[1])
end
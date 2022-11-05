using MLDatasets
using Images

using Flux, DiffEqFlux
using Flux.Data: DataLoader
using Flux: update!
using CUDA

include("dense_pos_layer.jl")

function get_data(batch_size)
    train_x, train_y = MNIST(split=:train)[:]
    train_x_flat = reshape(train_x, 28^2, :)
    DataLoader((train_x_flat, train_y), batchsize=batch_size, shuffle=true), train_x, train_y
end

function get_data_cuda(batch_size)
    train_x, train_y = gpu(MNIST(split=:train)[:])
    train_x_flat = reshape(train_x, 28^2, :)
    DataLoader((train_x_flat, train_y), batchsize=batch_size, shuffle=true), train_x, train_y
end

function run(epochs=1)
    dataloader, train_x, train_y = get_data(128)
    input_pos = zeros(Float32, 28^2, 2)
    for y in 0:27
        for x in 0:27
            input_pos[1 + x + y * 28, 1] = ((x/27f0) - 0.5f0) * 2f0
            input_pos[1 + x + y * 28, 2] = ((y/27f0) - 0.5f0) * 2f0
        end
    end

    l1 = DensePosLayer(input_pos, 28*14; activation = swish)
    l2 = DensePosLayer(l1.positions, 30; activation = swish)
    l3 = DensePosLayer(l2.positions, 28*14; activation = swish)
    l4 = DensePosLayer(l3.positions, 28*28; activation = sigmoid_fast)
    l5 = Dense(5 => 10)

    model = Chain(l1, l2, l3, l4)
    train(model, dataloader, epochs=epochs), train_x, train_y
end

function train(model, dataloader; epochs=1, opt=Adam())
    i = 1
    loss(x) = Flux.mse(model(x), x)
    ps = Flux.params(model)
    for _ in 1:epochs
        for (x, y) in dataloader
            i = i + 1
            # try
                if i % 20 == 0
                    println(loss(x))
                end
                grad = gradient(() -> loss(x), ps)
                update!(opt, ps, grad)

            # catch e
            #     if typeof(e) <: InterruptException
            #         return weights, biases
            #     end
            # end
    
        end
    end
    return model
end

function view_output(model, input::Vector)
    (reshape(model(input), 28, 28) .|> Gray)'
end

function view_output(model, input::Matrix)
    (reshape(model(reshape(input, 28*28)), 28, 28) .|> Gray)'
end

function view_average_output(model, samples=1000)
    sum([view_output(model, rand(28, 28)) for _ in 1:samples]) ./ samples
end

function view_sample(input::Matrix)
    (reshape(input, 28, 28) .|> Gray)'
end
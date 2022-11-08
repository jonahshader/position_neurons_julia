using MLDatasets
using Images

using Flux, DiffEqFlux
using Flux.Data: DataLoader
using Flux: update!
using CUDA

using Plots

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

function make_model()
    input_pos = zeros(Float32, 28^2, 2)
    range = 2
    for y in 0:27
        for x in 0:27
            input_pos[1 + x + y * 28, 1] = ((x/27f0) - 0.5f0) * 2f0 * range
            input_pos[1 + x + y * 28, 2] = ((y/27f0) - 0.5f0) * 2f0 * range
        end
    end
    l1 = DensePosLayer(input_pos, 28*4; activation = swish, std = [1f0, 1f0])
    l2 = DensePosLayer(l1.positions, 30; activation = swish, std = [1f0, 1f0])
    l3 = DensePosLayer(l2.positions, 28*4; activation = swish, std = [1f0, 1f0])
    l4 = DensePosLayer(l3.positions, input_pos; activation = sigmoid_fast)

    Chain(l1, l2, l3, l4)
end

function run(epochs=1; opt=Adam())
    dataloader, train_x, train_y = get_data(128)
    model = make_model()
    train(model, dataloader, epochs=epochs, opt=opt), train_x, train_y
end

function run_cuda(epochs=1; opt=Adam())
    dataloader, train_x, train_y = get_data_cuda(128)
    model = make_model() |> gpu
    train(model, dataloader, epochs=epochs, opt=opt), train_x, train_y
end

function train(model, dataloader; epochs=1, opt=Adam())
    i = 1
    penalty() = sum([sum(model[i].weights .^ 2) + sum(model[i].bias .^ 2) for i in 1:length(model)]) * 0.000002f0
    loss(x) = Flux.mse(model(x), x)# + penalty()
    ps = Flux.params(model[begin:end-1], model[end].weights, model[end].bias)
    # ps = Flux.params([m.positions for m in model])
    for _ in 1:epochs
        for (x, y) in dataloader
            i = i + 1
            # try
                if i % 20 == 0
                    println(loss(x))
                    pos_cpu = [model[1].positions, model[2].positions, model[3].positions, model[4].positions] |> cpu
                    scatter(pos_cpu[1][:, 1], pos_cpu[1][:, 2], markersize=2)
                    scatter!(pos_cpu[2][:, 1], pos_cpu[2][:, 2], markersize=2)
                    scatter!(pos_cpu[3][:, 1], pos_cpu[3][:, 2], markersize=2)
                    scatter!(pos_cpu[4][:, 1], pos_cpu[4][:, 2], markersize=2) |> display
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
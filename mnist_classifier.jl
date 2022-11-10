using MLDatasets
using Images

using Flux
using Flux.Data: DataLoader
using Flux: update!
using CUDA

using Plots

include("dense_pos_layer.jl")
include("mnist_helper.jl")

function make_model()
    input_pos = zeros(Float32, 28^2, 2)
    range = 2
    for y in 0:27
        for x in 0:27
            input_pos[1 + x + y * 28, 1] = ((x/27f0) - 0.5f0) * 2f0 * range
            input_pos[1 + x + y * 28, 2] = ((y/27f0) - 0.5f0) * 2f0 * range
        end
    end

    output_pos = zeros(Float32, 10, 2)
    range = 3
    for x in 0:9
        output_pos[x+1, 1] = ((x/9) - 0.5f0) * 2f0 * range
        output_pos[x+1, 2] = 0f0
    end
    l1 = DensePosLayer(input_pos, 28*4; activation = swish)
    l2 = DensePosLayer(l1.positions, 28*2; activation = swish)
    l3 = DensePosLayer(l2.positions, 30, activation = swish)
    l4 = DensePosLayer(l3.positions, output_pos)

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
    # loss(x) = Flux.mse(model(x), x) + penalty()
    loss(x, y) = Flux.logitcrossentropy(model(x), y) + penalty()
    ps = Flux.params(model[begin:end-1], model[end].weights, model[end].bias)
    # ps = Flux.params([m.positions for m in model])
    for _ in 1:epochs
        for (x, y) in dataloader
            i = i + 1
            # try
                if i % 20 == 0
                    println(loss(x, y))
                    pos_cpu = [model[1].positions, model[2].positions, model[3].positions, model[4].positions] |> cpu
                    scatter(pos_cpu[1][:, 1], pos_cpu[1][:, 2], markersize=2)
                    scatter!(pos_cpu[2][:, 1], pos_cpu[2][:, 2], markersize=2)
                    scatter!(pos_cpu[3][:, 1], pos_cpu[3][:, 2], markersize=2)
                    scatter!(pos_cpu[4][:, 1], pos_cpu[4][:, 2], markersize=2) |> display
                end
                grad = gradient(() -> loss(x, y), ps)
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

function accuracy(m, test_x, test_y)
    sum([reshape(test_x[:, :, i], 28^2) |> m |> argmax == (test_y[i]+1) for i in 1:length(test_y)]) / length(test_y)
end
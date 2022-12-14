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
    range = 5
    for x in 0:9
        # output_pos[x+1, 1] = ((x/9) - 0.5f0) * 2f0 * range
        # output_pos[x+1, 2] = 5f0
        p = (x / 10) * pi*2
        output_pos[x+1, 1] = cos(p) * range
        output_pos[x+1, 2] = sin(p) * range
    end
    l1 = DensePosLayer(input_pos, 28*4; activation = swish)
    l2 = DensePosLayer(l1.positions, 28*2; activation = swish, std=[1.5f0, 1.5f0])
    l3 = DensePosLayer(l2.positions, 50, activation = swish, std=[2f0, 2f0])
    l4 = DensePosLayer(l3.positions, 50, activation = swish, std=[2f0, 2f0])
    l5 = DensePosLayer(l4.positions, 50, activation = swish, std=[2f0, 2f0])
    l6 = DensePosLayer(l5.positions, output_pos)

    Chain(l1, l2, l3, l4, l5, l6)
end

function run(epochs=1; opt=Adam(), batch=128)
    dataloader, train_x, train_y = get_data(batch)
    model = make_model()
    train(model, dataloader, epochs=epochs, opt=opt), train_x, train_y
end

function run_cuda(epochs=1; opt=Adam(), batch=128, p=1.0, save_vis=false)
    dataloader, train_x, train_y = get_data_cuda_partial(batch, p)
    model = make_model() |> gpu
    train(model, dataloader, epochs=epochs, opt=opt, save_vis=save_vis), train_x, train_y
end

function train(model, dataloader; epochs=1, opt=Adam(), pos_regularization=true, save_vis=false)
    i = 1
    penalty() = sum([sum(model[i].weight .^ 2) + sum(model[i].bias .^ 2) for i in 1:length(model)]) * 0.00002f0
    dist_penalty() = sum([get_weighted_mean_distance2(model[i]) for i in 1:length(model)]) / (length(model) * 4)
    # loss(x) = Flux.mse(model(x), x) + penalty()
    loss_with_pos_regularization(x, y) = Flux.logitcrossentropy(model(x), y) + dist_penalty()
    regular_loss(x, y) = Flux.logitcrossentropy(model(x), y)
    loss = regular_loss
    if pos_regularization
        loss = loss_with_pos_regularization
    end
    ps = 
    if pos_regularization
        Flux.params(model[begin:end-1], model[end].weight, model[end].bias)
        # Flux.params(model[begin+1:end-1], model[begin].weight, model[begin].bias, model[end].weight, model[end].bias)
    else
        Flux.params(model)
    end
    # ps = Flux.params([m.positions for m in model[begin:end-1]])
    anim = Animation()
    for _ in 1:epochs
        for (x, y) in dataloader
            i = i + 1
            # try
                if i % 20 == 0
                    println(loss(x, y))
                    if pos_regularization
                        pos_cpu = vcat([model[begin].previous_positions], [model[i].positions for i in 1:length(model)]) |> cpu
                        scatter(pos_cpu[1][:, 1], pos_cpu[1][:, 2], markersize=1, seriesalpha=0.25)
                        for p in pos_cpu[begin+1:end-1]
                            scatter!(p[:, 1], p[:, 2], markersize=3)
                        end
                        scatter!(pos_cpu[end][:, 1], pos_cpu[end][:, 2], markersize=3) |> display
                        if save_vis
                            frame(anim)
                        end


                        
                    end
                end
                grad = gradient(() -> loss(x, y), ps)
                update!(opt, ps, grad)

            # catch e
            #     if typeof(e) <: InterruptException
            #         return weight, biases
            #     end
            # end
    
        end
    end
    if save_vis
        gif(anim, "Animation.gif", fps=30)
    end
    return model
end

function accuracy(m, test_x, test_y)
    sum([reshape(test_x[:, :, i], 28^2) |> m |> argmax == (test_y[i]+1) for i in 1:length(test_y)]) / length(test_y)
end

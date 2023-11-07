using MLDatasets
using CUDA
using Flux
using Images

function get_data(batch_size)
    train_x, train_y = MNIST(split=:train)[:]
    train_x_flat = reshape(train_x, 28^2, :)
    train_y_onehot = Flux.onehotbatch(train_y, 0:9)
    DataLoader((train_x_flat, train_y_onehot), batchsize=batch_size, shuffle=true), train_x, train_y
end

function get_data_cuda(batch_size, split=:train)
    train_x, train_y = MNIST(split=split)[:]
    train_x_flat = reshape(train_x, 28^2, :) |> gpu
    train_y_onehot = Flux.onehotbatch(train_y, 0:9) |> gpu
    DataLoader((train_x_flat, train_y_onehot), batchsize=batch_size, shuffle=true), train_x, train_y
end

function get_data_cuda_partial(batch_size, p)
    train_x, train_y = MNIST(split=:train)[:]
    selection = rand(size(train_y)...) .<= p
    train_x = train_x[:, :, selection]
    train_y = train_y[selection]
    train_x_flat = reshape(train_x, 28^2, :) |> gpu
    train_y_onehot = Flux.onehotbatch(train_y, 0:9) |> gpu
    DataLoader((train_x_flat, train_y_onehot), batchsize=batch_size, shuffle=true), train_x, train_y
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

function validate(model, input, output)
    input_flattened = Flux.flatten(input)
    sum([a[1] for a in vec(Tuple.(argmax(model(input_flattened), dims=1)))] .-1 .== output) / length(output)
end
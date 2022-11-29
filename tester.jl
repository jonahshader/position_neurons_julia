include("mnist_classifier.jl")

using StatsPlots

function run_tests(run_function::Function, configurations)
    return [run_function(c) for c in configurations]
end

function run_repeated(run_function::Function, config, trials)
    return [run_function(config) for _ in 1:trials]
end

# this takes a config and returns the performance (validation)
function mnist_denseposlayer_classifier_run(config)
    proportion, batch_size, epochs, prune_proportion = config
    trained_model, _, _ = run_cuda(epochs; batch=batch_size, p=proportion) |> cpu
    prune_neurons!(trained_model, prune_proportion)
    test_x, test_y = MNIST(split=:test)[:] |> cpu

    validate(trained_model, test_x, test_y)
end

# this is similar to run_cuda from mnist_classifier.jl, but it uses a normal feedforward model
function run_vanilla_cuda(epochs=1; opt=Adam(), batch=128, p=1.0)
    dataloader, _, _ = get_data_cuda_partial(batch, p)
    model = Chain(
        Dense(28^2=>28*4, swish),
        Dense(28*4=>28*2, swish),
        Dense(28*2=>50, swish),
        Dense(50=>50, swish),
        Dense(50=>50, swish),
        Dense(50=>10)
    ) |> gpu
    train(model, dataloader, epochs=epochs, opt=opt, pos_regularization=false)
end

# this takes a config and returns the performance (validation)
function mnist_vanilla_classifier_run(config)
    proportion, batch_size, epochs, prune_proportion = config
    trained_model = run_vanilla_cuda(epochs; batch=batch_size, p=proportion) |> cpu
    test_x, test_y = MNIST(split=:test)[:] |> cpu
    prune_neurons!(trained_model, prune_proportion)
    validate(trained_model, test_x, test_y)
end

function config_and_run_tests(run_func, trials=3, proportions=10, min_dataset_proportion=0.25, max_dataset_proportion=0.25, min_prune_proportion=0.75, max_prune_proportion=0.75, batch_size=256, epochs=10)
    configs = [
        (
            (i * (max_dataset_proportion-min_dataset_proportion) / proportions) + min_dataset_proportion,
            batch_size,
            epochs,
            (i * (max_prune_proportion-min_prune_proportion) / proportions) + min_prune_proportion
        ) for i in 1:proportions]

    # run_tests(c -> sum(run_repeated(run_func, c, repeats)) / repeats, configs)
    run_tests(c -> run_repeated(run_func, c, trials), configs)
end

function create_histogram_of_weight(model)
    reduce(vcat, ([vec(l.weight) for l in model])) |> histogram
end

function average_trials(vec_of_trials)
    [sum(t)/length(t) for t in vec_of_trials]
end
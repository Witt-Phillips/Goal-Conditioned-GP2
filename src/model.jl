include("kernel.jl")
include("covariance.jl")
include("kernel_prior.jl")

using Gen

@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

# Full model
@gen function model(xs::Vector{Float64})
    
    # Generate a covariance kernel
    covariance_fn = {:tree} ~ covariance_prior()
    
    # Sample a global noise level
    noise ~ gamma_bounded_below(1, 1, 0.01)
    
    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    
    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys ~ mvnormal(zeros(length(xs)), cov_matrix)
    
    # Return the covariance function, for easy printing.
    return covariance_fn
end

function visualize_gp_trace(tr, xmin, xmax; title="")
    curveXs = collect(Float64, range(xmin, length=100, stop=xmax))
    data_xs, = get_args(tr)
    data_ys = tr[:ys]
    curveYs = [predict_ys(get_retval(tr), 0.000001, data_xs, data_ys, curveXs) for i=1:50]
    fig = plot()
    for (i, curveYSet) in enumerate(curveYs)
        plot!(curveXs, curveYSet, title=title, xlims=(xmin, xmax), label=nothing, color="lightgreen")
    end
    scatter!(data_xs, data_ys, color="black", label=nothing)
end

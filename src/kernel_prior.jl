include("kernel.jl")
using Gen

kernel_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]
@dist choose_kernel_type() = kernel_types[categorical([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])];

# Prior on kernels
@gen function covariance_prior()
    # Choose a type of kernel
    kernel_type ~ choose_kernel_type()

    # If this is a composite node, recursively generate subtrees
    if in(kernel_type, [Plus, Times])
        return kernel_type({:left} ~ covariance_prior(), {:right} ~ covariance_prior())
    end
    
    # Otherwise, generate parameters for the primitive kernel.
    kernel_args = (kernel_type == Periodic) ? [{:scale} ~ uniform(0, 1), {:period} ~ uniform(0, 1)] : [{:param} ~ uniform(0, 1)]
    return kernel_type(kernel_args...)
end;

# sanity check; look at some sampled programs
function sample_10_kernels()
    for i=1:10
        println(covariance_prior())
    end
end

# sample_10_kernels()
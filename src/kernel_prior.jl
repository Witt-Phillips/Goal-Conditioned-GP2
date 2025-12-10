# = Adapted from the Gen tutorials: https://www.gen.dev/tutorials/rj/tutorial#synthesis=#

include("kernel.jl")
using Gen

kernel_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]

# equal prob
@dist choose_kernel_type() = kernel_types[categorical([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])];

# Prior on kernels
@gen function covariance_prior()
    kernel_type ~ choose_kernel_type()

    # recursive subtree generation
    if in(kernel_type, [Plus, Times])
        return kernel_type({:left} ~ covariance_prior(), {:right} ~ covariance_prior())
    end
    
    # generate parameters for the primitive kernel
    if kernel_type == Periodic
        # favor smaller periods to get more repeated patterns
        scale = ({:scale} ~ uniform(0, 1))
        period = ({:period} ~ uniform(0.05, 0.3))
        return Periodic(scale, period)
    else
        param = ({:param} ~ uniform(0, 1))
        return kernel_type(param)
    end
end;
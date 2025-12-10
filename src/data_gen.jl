# DSL for generating trees of functions
# Similar structure to kernel.jl but for generating data functions

"""Node in a tree representing a function"""
abstract type FunctionNode end
abstract type PrimitiveFunction <: FunctionNode end
abstract type CompositeFunction <: FunctionNode end

"""
    size(::FunctionNode)
Number of nodes in the tree describing this function.
"""
Base.size(::PrimitiveFunction) = 1
Base.size(node::CompositeFunction) = node.size

"""Linear function: slope * x + intercept"""
struct Lin <: PrimitiveFunction
    slope::Float64
    intercept::Float64
end

"""Evaluate linear function at point x"""
function eval_func(node::Lin, x::Float64)
    node.slope * x + node.intercept
end

"""Evaluate linear function at vector of points"""
function eval_func(node::Lin, xs::Vector{Float64})
    node.slope .* xs .+ node.intercept
end


"""Sine function: amplitude * sin(2π * x / period)"""
struct Sin <: PrimitiveFunction
    amplitude::Float64
    period::Float64
end

"""Evaluate sine function at point x"""
function eval_func(node::Sin, x::Float64)
    node.amplitude * sin(2 * π * x / node.period)
end

"""Evaluate sine function at vector of points"""
function eval_func(node::Sin, xs::Vector{Float64})
    node.amplitude .* sin.(2 .* π .* xs ./ node.period)
end


"""Add node: left + right"""
struct Add <: CompositeFunction
    left::FunctionNode
    right::FunctionNode
    size::Int
end

Add(left, right) = Add(left, right, size(left) + size(right) + 1)

"""Evaluate addition at point x"""
function eval_func(node::Add, x::Float64)
    eval_func(node.left, x) + eval_func(node.right, x)
end

"""Evaluate addition at vector of points"""
function eval_func(node::Add, xs::Vector{Float64})
    eval_func(node.left, xs) .+ eval_func(node.right, xs)
end


"""Multiply node: left * right"""
struct Mul <: CompositeFunction
    left::FunctionNode
    right::FunctionNode
    size::Int
end

Mul(left, right) = Mul(left, right, size(left) + size(right) + 1)

"""Evaluate multiplication at point x"""
function eval_func(node::Mul, x::Float64)
    eval_func(node.left, x) * eval_func(node.right, x)
end

"""Evaluate multiplication at vector of points"""
function eval_func(node::Mul, xs::Vector{Float64})
    eval_func(node.left, xs) .* eval_func(node.right, xs)
end


using Gen

function_types = [Lin, Sin, Add, Mul]
@dist choose_function_type() = function_types[categorical([0.3, 0.3, 0.2, 0.2])];

# Prior on function trees
@gen function function_prior()
    # Choose a type of function
    function_type ~ choose_function_type()

    # If this is a composite node, recursively generate subtrees
    if in(function_type, [Add, Mul])
        return function_type({:left} ~ function_prior(), {:right} ~ function_prior())
    end
    
    # Otherwise, generate parameters for the primitive function.
    if function_type == Lin
        slope = ({:slope} ~ uniform(-1.0, 1.0))
        intercept = ({:intercept} ~ uniform(0, 1.0))
        return Lin(slope, intercept)
    elseif function_type == Sin
        amplitude = ({:amplitude} ~ uniform(0.1, 2.0))
        period = ({:period} ~ uniform(0.1, 2.0))
        return Sin(amplitude, period)
    end
end;

"""Normalize values to [0, 1] range"""
function normalize(vals::Vector{Float64})
    min_val = minimum(vals)
    max_val = maximum(vals)
    if max_val == min_val
        return fill(0.5, length(vals))  # All same value, return middle
    end
    return (vals .- min_val) ./ (max_val - min_val)
end

"""Generate data points from a function with random x range, then normalize to [0,1]"""
function get_data(fn::FunctionNode, n_points=21; noise=0.0)
    
    # Sample x range from prior (e.g., uniform over some reasonable range)
    xmin = rand() * 4.0 - 2.0  # Range from -2 to 2
    xmax = xmin + (rand() * 4.0 + 0.5)  # Width between 0.5 and 4.5
    
    # Generate points in the sampled range
    xs_raw = collect(Float64, range(xmin, length=n_points, stop=xmax))
    ys_raw = eval_func(fn, xs_raw)
    
    # Add noise if specified (Gaussian noise with standard deviation = noise)
    if noise > 0.0
        ys_raw = ys_raw .+ randn(n_points) .* noise
    end
    
    # Normalize both x and y to [0, 1]
    xs = normalize(xs_raw)
    ys = normalize(ys_raw)
    
    return xs, ys, fn, (xmin, xmax)
end


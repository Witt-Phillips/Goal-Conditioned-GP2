#= DSL for generating trees of functions =#
# similiar structure to kernel.jl but for generating data functions

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
using JSON
using Dates

function_types = [Lin, Sin, Add, Mul]
@dist choose_function_type() = function_types[categorical([0.25, 0.25, 0.25, 0.25])];

# Prior on function trees
@gen function function_prior()
    function_type ~ choose_function_type()

    # recursive subtree generation
    if in(function_type, [Add, Mul])
        return function_type({:left} ~ function_prior(), {:right} ~ function_prior())
    end
    
    if function_type == Lin
        slope = ({:slope} ~ uniform(-1.0, 1.0))
        intercept = ({:intercept} ~ uniform(0, 1.0))
        return Lin(slope, intercept)
    elseif function_type == Sin
        amplitude = ({:amplitude} ~ uniform(0.1, 2.0))
        period = ({:period} ~ uniform(0.05, 0.5))
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
    
    xmin = rand() * 4.0 - 2.0  # Range from -2 to 2
    xmax = xmin + (rand() * 4.0 + 0.5)  # Width between 0.5 and 4.5
    
    xs_raw = collect(Float64, range(xmin, length=n_points, stop=xmax))
    ys_raw = eval_func(fn, xs_raw)
    
    if noise > 0.0
        ys_raw = ys_raw .+ randn(n_points) .* noise
    end
    
    xs = normalize(xs_raw)
    ys = normalize(ys_raw)
    
    return xs, ys, fn, (xmin, xmax)
end

# ============================================================================
# Save/Load functionality for interesting data runs
# ============================================================================

"""Serialize a FunctionNode to a dictionary"""
function serialize_function(fn::FunctionNode)
    if fn isa Lin
        return Dict(
            "type" => "Lin",
            "slope" => fn.slope,
            "intercept" => fn.intercept
        )
    elseif fn isa Sin
        return Dict(
            "type" => "Sin",
            "amplitude" => fn.amplitude,
            "period" => fn.period
        )
    elseif fn isa Add
        return Dict(
            "type" => "Add",
            "left" => serialize_function(fn.left),
            "right" => serialize_function(fn.right)
        )
    elseif fn isa Mul
        return Dict(
            "type" => "Mul",
            "left" => serialize_function(fn.left),
            "right" => serialize_function(fn.right)
        )
    else
        error("Unknown function type: $(typeof(fn))")
    end
end

"""Deserialize a dictionary back to a FunctionNode"""
function deserialize_function(d::Dict)
    type = d["type"]
    if type == "Lin"
        return Lin(d["slope"], d["intercept"])
    elseif type == "Sin"
        return Sin(d["amplitude"], d["period"])
    elseif type == "Add"
        return Add(deserialize_function(d["left"]), deserialize_function(d["right"]))
    elseif type == "Mul"
        return Mul(deserialize_function(d["left"]), deserialize_function(d["right"]))
    else
        error("Unknown function type: $type")
    end
end

"""Save a data run to a JSON file"""
function save_data_run(xs, ys, fn, xrange, name::String=""; 
                       n_points::Int=length(xs), noise::Float64=0.0,
                       description::String="")
    # Create data directory if it doesn't exist
    data_dir = "data"
    if !isdir(data_dir)
        mkdir(data_dir)
    end
    
    # Generate filename
    if name == ""
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = joinpath(data_dir, "run_$timestamp.json")
    else
        # Sanitize name for filename
        safe_name = replace(name, r"[^a-zA-Z0-9_-]" => "_")
        filename = joinpath(data_dir, "run_$safe_name.json")
    end
    
    # Prepare data structure
    xmin, xmax = xrange
    data = Dict(
        "xs" => xs,
        "ys" => ys,
        "function" => serialize_function(fn),
        "xmin" => xmin,
        "xmax" => xmax,
        "n_points" => n_points,
        "noise" => noise,
        "description" => description,
        "saved_at" => string(now()),
        "name" => name == "" ? basename(filename) : name
    )
    
    # Write to file
    open(filename, "w") do f
        JSON.print(f, data, 2)
    end
    
    println("Saved data run to: $filename")
    return filename
end

"""Load a data run from a JSON file"""
function load_data_run(filename::String)
    # If just a name is provided, try to find it in data directory
    if !occursin("/", filename) && !occursin("\\", filename)
        data_dir = "data"
        if isdir(data_dir)
            # Try exact match first
            full_path = joinpath(data_dir, filename)
            if !endswith(filename, ".json")
                full_path = joinpath(data_dir, "$filename.json")
            end
            if isfile(full_path)
                filename = full_path
            else
                # Try to find by name pattern
                pattern = occursin(".json", filename) ? filename : "$filename.json"
                matches = filter(f -> occursin(pattern, f), readdir(data_dir))
                if length(matches) == 1
                    filename = joinpath(data_dir, matches[1])
                elseif length(matches) > 1
                    error("Multiple files match '$pattern'. Please specify full filename.")
                else
                    error("File not found: $filename")
                end
            end
        end
    end
    
    # Read and parse JSON
    data = JSON.parsefile(filename)
    
    # Deserialize function
    fn = deserialize_function(data["function"])
    
    # Extract data
    xs = Vector{Float64}(data["xs"])
    ys = Vector{Float64}(data["ys"])
    xmin = Float64(data["xmin"])
    xmax = Float64(data["xmax"])
    
    println("Loaded data run: $(get(data, "name", basename(filename)))")
    if haskey(data, "description") && data["description"] != ""
        println("  Description: $(data["description"])")
    end
    if haskey(data, "saved_at")
        println("  Saved at: $(data["saved_at"])")
    end
    
    return xs, ys, fn, (xmin, xmax)
end

"""List all saved data runs"""
function list_saved_runs()
    data_dir = "data"
    if !isdir(data_dir)
        println("No data directory found. No saved runs.")
        return []
    end
    
    runs = filter(f -> startswith(f, "run_") && endswith(f, ".json"), readdir(data_dir))
    
    if length(runs) == 0
        println("No saved runs found.")
        return []
    end
    
    println("Saved data runs:")
    println("=" ^ 60)
    
    for (i, run_file) in enumerate(runs)
        try
            data = JSON.parsefile(joinpath(data_dir, run_file))
            name = get(data, "name", run_file)
            desc = get(data, "description", "")
            saved_at = get(data, "saved_at", "unknown")
            n_points = get(data, "n_points", "?")
            noise = get(data, "noise", "?")
            
            println("$i. $name")
            if desc != ""
                println("   Description: $desc")
            end
            println("   File: $run_file")
            println("   Points: $n_points, Noise: $noise")
            println("   Saved: $saved_at")
            println()
        catch e
            println("$i. $run_file (error reading: $e)")
            println()
        end
    end
    
    return runs
end


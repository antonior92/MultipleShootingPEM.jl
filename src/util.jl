# Sparse Matrix COO
struct SparseMatrixCOO{Tv, Ti}
    m::Int
    n::Int
    i::Vector{Ti}
    j::Vector{Ti}
    v::Vector{Tv}
end


# Linear Operator
struct LinearOperator
    shape::Tuple{Int, Int}
    matvec::Function
end

function Base.dot(op::LinearOperator, x::Vector{Float64})
    return op.matvec(x)
end

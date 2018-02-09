# Sparse Matrix COO
struct SparseMatrixCOO{Tv, Ti}
    m::Int
    n::Int
    i::Vector{Ti}
    j::Vector{Ti}
    v::Vector{Tv}
end

to_sparse_csc(coo::SparseMatrixCOO) = sparse(coo.i, coo.j, coo.v, coo.m, coo.n)

to_full(coo::SparseMatrixCOO) = full(to_sparse_csc(coo))

to_python(coo::SparseMatrixCOO) = scipy_sps["coo_matrix"](
    (coo.v, (coo.i-1, coo.j-1)), shape=(coo.m, coo.n))

# Linear Operator
struct LinearOperator
    m::Int
    n::Int
    matvec::Function
end

function Base.dot(op::LinearOperator, x::Vector{Float64})
    return op.matvec(x)
end

struct SparseMatrixCOO{Tv, Ti}
    m::Int
    n::Int
    i::Vector{Ti}
    j::Vector{Ti}
    v::Vector{Tv}
end

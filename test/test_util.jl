@testset "Test SparseMatrixCOO" begin
    i = Int[1, 4, 2, 1]
    j = Int[1, 4, 2, 3]
    v = Float64[4, 5, 7, 8]
    coo = ms.SparseMatrixCOO(4, 4, i, j, v)

    @test ms.to_sparse_csc(coo) == Float64[4 0 8 0;
                                           0 7 0 0;
                                           0 0 0 0;
                                           0 0 0 5]
    @test ms.to_full(coo) == Float64[4 0 8 0;
                                     0 7 0 0;
                                     0 0 0 0;
                                     0 0 0 5]

    coo_py = ms.to_python(coo)

    @test coo_py["dot"]([1, 2, 3, 4]) == Float64[28, 14, 0, 20]
    @test coo_py["toarray"]() == Float64[4 0 8 0;
                                         0 7 0 0;
                                         0 0 0 0;
                                         0 0 0 5]
end

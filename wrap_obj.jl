include("f_obj_kernel.jl")

function _obj!()

    @cuda threads=threads blocks=blocks shmem=shmem _Btot!(fld, B,                                  # Valores base y alocaciones
                grid.X, grid.Y, grid.Z,                                                             # Grid de evaluación
                P, Θ, M, m, N)                                                                      # Pos y θ de vec momento dipolo
    
    @cuda threads=threads blocks=blocks _grad!(B, Gx, Gy, Gz, dy_m, grid.nx, grid.ny, grid.nz, N)
    
    @cuda threads=threads blocks=blocks shmem=shmem_bytes _metrics!(B, by_min, by_max, grad_rms, Gx, Gy, Gz, msk, N, Nmsk)        

    return #w * (by_max - by_min) + λ * sqrt.(grad_rms)
end
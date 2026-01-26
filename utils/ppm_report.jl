include("../kernels/f_kernel.jl")


function get_ppm_RMS(bestθ, λ )
    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ )
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)
    return 1000000 * (by_max - by_min) / by_mean
end


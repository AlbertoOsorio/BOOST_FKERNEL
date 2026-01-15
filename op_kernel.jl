function _mutate!(θ, step_deg, lo, hi, Nmags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end

    if rand() < 0.3
        θ[idx] += step_deg * randn()
        θ[idx] = min(max(θ[idx], lo[idx]), hi[idx])
    end
    return

end
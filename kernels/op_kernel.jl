
using GPUArrays: @allowscalar

function _mutate!(θ, step_deg, lo, hi, Nmags, tresh)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end

    @inbounds if rand() < tresh
        θ[idx] += step_deg * randn()
        θ[idx] = min(max(θ[idx], lo[idx]), hi[idx])
    end
    return

end

function _reset!(θ, lo, hi, Nmags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end

    @inbounds θ[idx] = lo[idx] + rand() * (hi[idx] - lo[idx])
    return
end

@inline function cool(t0, alpha, k)
    return t0 * (alpha^k)
end


function overwrite_cuarray!(x, xop)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  

    N = length(x)
    if idx > N; return; end

    @inbounds xop[idx] = x[idx]
    return
end

function naive_SA_RMS!(f, λ; lower=lower, upper=upper,
                     iters=4_000, restarts=5,
                     T0=0.1, alpha=0.995, step0=10.0, step_min=0.5,
                     report_every=50)
    
    lo = CuArray(lower)
    hi = CuArray(upper)

    f(θ_init, λ)
    bestθ_global = copy(θ_init)
    bestf_global = CuArray([0.0f0])
    @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(coef, bestf_global)

    Θ = copy(bestθ_global)

    for r in 1:restarts
        if r == 1
        else
            @cuda threads=512 blocks=cld(336, 512) _reset!(Θ, lo, hi, 336)
        end

        f(Θ, λ)

        step = step0
        T = T0
        step_decay = (step0 > step_min) ? (step_min/step0)^(1/iters) : 1.0

        for k in 1:iters
    
            @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(Θ, Θmew)          # copia el valor de θ en Θmew
            @cuda threads=512 blocks=cld(336, 512) _mutate!(Θmew, step, lo, hi, 336, 0.3)
            
            @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(coef, f_prev)

            f(Θmew, λ)

            @allowscalar if (coef[1] < f_prev[1]) || (rand() < exp(-(coef[1] - f_prev[1])/max(T,1e-9)))
                @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(Θmew, Θ)
                @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(coef, f_prev)
                if coef[1] < bestf_global[1]
                    @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(coef, bestf_global)
                    @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(Θ, bestθ_global)
                end
            end

            T = cool(T0, alpha, k)
            step = max(step*step_decay, step_min)
            @allowscalar begin
                if (k % report_every == 0)
                    @info "SA restart=$r iter=$k  f=$coef best=$bestf_global  T=$(round(T,digits=4))  step=$(round(step,digits=2))"
                end
            end
        end
    end
    return bestθ_global
end

function naive_SA_STDIV!(f; lower=lower, upper=upper,
                     iters=2_000, restarts=5,
                     T0=0.1, alpha=0.995, step0=10.0, step_min=0.5,
                     report_every=50)
    
    lo = CuArray(lower)
    hi = CuArray(upper)

    bestf_global = copy(stdiv)
    bestθ_global = copy(θ_init)
    Θ = copy(bestθ_global)
    
    f(Θ)

    @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(stdiv, bestf_global)


    for r in 1:restarts
        if r == 1
        else
            @cuda threads=512 blocks=cld(336, 512) _reset!(Θ, lo, hi, 336)
        end

        f(Θ)

        step = step0
        T = T0
        step_decay = (step0 > step_min) ? (step_min/step0)^(1/iters) : 1.0

        for k in 1:iters
    
            @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(Θ, Θmew)          # copia el valor de θ en Θmew
            @cuda threads=512 blocks=cld(336, 512) _mutate!(Θmew, step, lo, hi, 336, 0.3)
            
            @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(stdiv, f_prev)

            f(Θmew)

            @allowscalar if (stdiv[1] < f_prev[1]) || (rand() < exp(-(stdiv[1] - f_prev[1])/max(T,1e-9)))
                @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(Θmew, Θ)
                @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(stdiv, f_prev)
                if stdiv[1] < bestf_global[1]
                    @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(stdiv, bestf_global)
                    @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(Θ, bestθ_global)
                end
            end

            T = cool(T0, alpha, k)
            step = max(step*step_decay, step_min)
            @allowscalar begin
                if (k % report_every == 0)
                    @info "SA restart=$r iter=$k  f=$stdiv best=$bestf_global  T=$(round(T,digits=4))  step=$(round(step,digits=2))"
                end
                @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(CuArray([0.0f0]), stdiv)
                @cuda threads=512 blocks=cld(336, 512) overwrite_cuarray!(CuArray([0.0f0]), by_mean)
            end

            

        end
    end
    return bestθ_global
end
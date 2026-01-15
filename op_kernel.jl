


function _mutate!(θ, step_deg, lo, hi, Nmags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end

    if rand() < 0.3
        θ[idx] += step_deg * randn()
        θ[idx] = min(max(θ[idx], lo[idx]), hi[idx])
    end
    return

end

@inline function cool(t0, alpha, k)
    return t0 * (alpha^k)
end

function naive_SA!(f; lower=lower, upper=upper,
                     iters=2_000,
                     T0=0.1, alpha=0.995, step0=10.0, step_min=0.5,
                     report_every=50)

    fθ = f(θ)
    step = step0
    T = T0
    step_decay = (step0 > step_min) ? (step_min/step0)^(1/iters) : 1.0

    for k in 1:iters
        θnew = copy(θ)
        @cuda threads=512 blocks=cld(336, 512) _mutate!(θnew, step, lower, upper, 336)
        fnew = f(θnew)

        if (fnew < fθ) || (rand() < exp(-(fnew - fθ)/max(T,1e-9)))
            θ .= θnew
            fθ = fnew
            if fθ < bestf_global
                bestf_global = fθ
                bestθ_global .= θ
            end
        end

        T = cool(T0, alpha, k)
        step = max(step*step_decay, step_min)

        if (k % report_every == 0)
            @info "SA restart=$r iter=$k  f=$fθ  best=$bestf_global  T=$(round(T,digits=4))  step=$(round(step,digits=2))"
        end
    end
end
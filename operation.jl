include("setup.jl")
include("utils/wrap.jl")
include("utils/ppm_report.jl")


function generate_Lcurve()
    for baseline = 1:10
        λ = baseline / 10
        bestθ, bestf =  naive_SA_RMS!(RMS_operation, λ)
        ppm = get_ppm_RMS(bestθ, λ)
        @save "data/test00/lambdaeq_$λ.jld2" λ bestθ bestf ppm
    end
end


λ = 0.5
bestθ =  naive_SA_RMS!(RMS_operation, λ)
ppm = get_ppm_RMS(bestθ, λ)
@save "data/test00/lambdaeq_$λ.jld2" λ bestθ ppm
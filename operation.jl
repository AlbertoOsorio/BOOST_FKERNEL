include("setup.jl")
include("utils/wrap.jl")
include("utils/ppm_report.jl")


for baseline = 1:10
    λ = baseline / 10
    bestθ, bestf =  naive_SA_RMS!(RMS_operation, λ)
    ppm = get_ppm_RMS(bestθ)
    @save "data/test00/lambdaeq_$λ.jld2" λ bestθ bestf ppm
end
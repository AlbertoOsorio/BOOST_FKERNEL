include("setup.jl")
include("utils/wrap.jl")
include("utils/ppm_report.jl")


function generate_Lcurve()
    for baseline = 1:10
        λ = baseline / 10
        bestθ, final_state =  naive_SA_RMS!(RMS_operation, λ)
        ppm = get_ppm_RMS(bestθ, λ, final_state)
        @save "data/test00/lambdaeq_$λ.jld2" λ bestθ final_state ppm
    end
end


λ = 0.5
bestθ, final_state =  naive_SA_RMS!(RMS_operation, λ)
ppm = get_ppm_RMS(bestθ, λ, final_state)
@save "data/test00/lambdaeq_$λ.jld2" λ bestθ final_state ppm
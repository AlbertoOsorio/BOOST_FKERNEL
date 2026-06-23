include("utils/wrap.jl")
include("utils/ppm_report.jl")
include("setup.jl")


function generate_Lcurve(T, alph)
    for baseline = 1:100
        include("setup.jl")
        λ = baseline / 100
        bestθ, final_state =  naive_SA_RMS!(RMS_operation, λ, vcat(-6:-1, 1:6), T0=T, alpha=alph)
        ppm = get_ppm_RMS(bestθ, λ, final_state)
        mkpath(string("runs/L_curve_150_params_T100$(round(Int, alph * 1000))/"))
        @save "runs/L_curve_150_params_T100$(round(Int, alph * 1000))/lambdaeq_$(round(Int, λ * 100)).jld2" λ bestθ final_state ppm
    end
end

function optim_run()
    save_name = "single_ring_seq"
    file_name = "best_from_SR_seq.jld2"

    λ = 0.0
    test_ring_seq = [2,3,1,4]
    bestθ, final_state =  naive_SA_RMS!(RMS_operation, λ, test_ring_seq, T0=100, alpha=0.999)
    ppm = get_ppm_RMS(bestθ, λ, final_state, save_name)

    mkpath(string("runs/", save_name))
    @save string("runs/$save_name/", file_name) λ bestθ final_state ppm
end

function optim_run_SD()
    save_name = "single_ring_seq"
    file_name = "best_from_SR_seq.jld2"

    λ = 0.0
    test_ring_seq = [2,3,1,4]
    bestθ, final_state = naive_SA_STDIV!(STDIV_operation, test_ring_seq, T0=100, alpha=0.999)
    ppm = get_ppm_RMS(bestθ, λ, final_state, save_name)

    mkpath(string("runs/", save_name))
    @save string("runs/$save_name/", file_name) λ bestθ final_state ppm
end



# a = [0.85, 0.87, 0.89, 0.9, 0.92, 0.95, 0.96, 0.97, 0.99, 0.999]

# for alph in a
#     generate_Lcurve(0.1, alph)
# end

#λ = 0.3
#bestθ, final_state =  naive_SA_RMS!(RMS_operation, λ)
#ppm = get_ppm_RMS(bestθ, λ, final_state)
#@save "data/L_curve_150/lambdaeq_$λ.jld2" λ bestθ final_state ppm
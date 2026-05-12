using FileIO
include("../kernels/f_kernel.jl")


function get_ppm_RMS(bestθ, λ, final_state)
    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ, final_state)
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    B_res = Array(B)

    for i = 1:11
       slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Create a copy for plotting and apply the mask
        # We replace 0s (or whatever your 'masked' value is) with NaN
        plot_data = copy(slice)
        plot_data[mask_slice .== 0] .= NaN 

        fig = Figure(resolution = (800, 600))

        ax = Axis(fig[1, 1];
            title = "Fieldmap slice z = $(i)",
            xlabel = "x mm",
            ylabel = "y mm"
        )

        # Use nan_color to make the masked areas black
        hm = heatmap!(ax, xg, yg, plot_data; 
            colormap = :viridis,
            colorrange = (46.1, 47.0),
            nan_color = :black  # This turns the NaNs black
        )
        
        Colorbar(fig[1, 2], hm, label = "mT")

        dir = string("runs/L_curve_150_params01/imgs/$(round(Int, λ * 100))/")
        mkpath(dir) 

        save(string("runs/L_curve_150_params01/imgs/$(round(Int, λ * 100))/","$(i)_slice_at_$(round(Int, λ * 100)).png"), fig)
    end

    return 1000000 * (by_max - by_min) / by_mean
end


function graph_form_jld2()

    f_data = "runs/L_curve_150_params_T100999/lambdaeq_8.jld2"
    @load f_data  λ bestθ final_state ppm

    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ, final_state)
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    B_res = Array(B)

    for i = 1:11
       slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Create a copy for plotting and apply the mask
        # We replace 0s (or whatever your 'masked' value is) with NaN
        plot_data = copy(slice)
        plot_data[mask_slice .== 0] .= NaN 

        fig = Figure(resolution = (800, 600))

        ax = Axis(fig[1, 1];
            title = "Fieldmap slice z = $(i)",
            xlabel = "x mm",
            ylabel = "y mm"
        )

        # Use nan_color to make the masked areas black
        hm = heatmap!(ax, xg, yg, plot_data; 
            colormap = :viridis,
            colorrange = (46.1, 47.0),
            nan_color = :black  # This turns the NaNs black
        )
        
        Colorbar(fig[1, 2], hm, label = "mT")

        dir = string("runs/L_curve_150_params_T100999/imgs/")
        mkpath(dir) 

        save(string("runs/L_curve_150_params_T100999/imgs/","$(i)_slice.png"), fig)
    end

end

function ppm_slice()

    f_data = "runs/L_curve_150_params_T100999/lambdaeq_8.jld2"
    @load f_data  λ bestθ final_state ppm

    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ, final_state)
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    B_res = Array(B)

    ppm_per_slice = []

    for i = 1:11
        slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Create a copy for plotting and apply the mask
        # We replace 0s (or whatever your 'masked' value is) with NaN
        data = copy(slice)
        data[mask_slice .== 0] .= NaN 
        min_val = minimum(filter(!isnan, data))
        max_val = maximum(filter(!isnan, data))
        μ = mean(x for x in data if !isnan(x))

        push!(ppm_per_slice, (max_val - min_val)/μ)

    end
    return ppm_per_slice
end


function ppm_slice_ori()

    filee = "data/parallelepiped_150x150x150.jld2"
    @load filee  By_3D xg yg zg

    B_res = By_3D

    ppm_per_slice = []

    for i = 1:11
        slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Create a copy for plotting and apply the mask
        # We replace 0s (or whatever your 'masked' value is) with NaN
        data = copy(slice)
        data[mask_slice .== 0] .= NaN 
        min_val = minimum(filter(!isnan, data))
        max_val = maximum(filter(!isnan, data))
        μ = mean(x for x in data if !isnan(x))

        push!(ppm_per_slice, (max_val - min_val)/μ)

    end
    return ppm_per_slice
end

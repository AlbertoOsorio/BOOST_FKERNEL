using FileIO
include("../kernels/f_kernel.jl")


function get_ppm_RMS(bestθ, λ, final_state)
    by_mean = CuArray([0.0f0])
    RMS_operation(bestθ, λ, final_state)
    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    B_res = Array(B)

    for i = 1:35
       slice = B_res[:,:,i]
        mask_slice = dmask[:,:,i]

        # Create a copy for plotting and apply the mask
        # We replace 0s (or whatever your 'masked' value is) with NaN
        plot_data = copy(slice)
        plot_data[mask_slice .== 0] .= NaN 

        val_min = minimum(x -> isnan(x) ? Inf : x, plot_data)
        val_max = maximum(x -> isnan(x) ? -Inf : x, plot_data)
        val_mean = mean(filter(!isnan, plot_data))

        ppm_slice = 1000000*(val_max - val_min)/val_mean

        fig = Figure(resolution = (800, 600))

        ax = Axis(fig[1, 1];
            title = "Fieldmap slice z = $(i) with local ppm = $(round(ppm_slice, digits=2))",
            xlabel = "x mm",
            ylabel = "y mm"
        )

        # Use nan_color to make the masked areas black
        hm = heatmap!(ax, xg, yg, plot_data; 
            colormap = :viridis,
            colorrange = (val_min,  val_max),
            nan_color = :black  # This turns the NaNs black
        )
        
        Colorbar(fig[1, 2], hm, label = "mT")

        dir = string("runs/L_curve_170/imgs/$(round(Int, λ * 100))/")
        mkpath(dir) 

        save(string("runs/L_curve_170/imgs/$(round(Int, λ * 100))/","$(i)_slice_at_$(round(Int, λ * 100)).png"), fig)
    end

    return 1000000 * (by_max - by_min) / by_mean
end
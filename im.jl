include("setup.jl")


@load "data/B.jld2" B

B_res = Array(B)

slice = B_res[:,:,60]

fig = Figure(resolution = (800, 600))

ax = Axis(fig[1, 1];
    title = "Fieldmap slice z = 60 pre shimming",
    xlabel = "x mm",
    ylabel = "y mm"
)

hm = heatmap!(ax, xg, yg, slice; colormap = :viridis,colorrange = (45, 45.3))
Colorbar(fig[1, 2], hm, label = "mT")  # o mT, ppm, lo que corresponda
fig

save(string("imgs/","fld360.png"), fig)
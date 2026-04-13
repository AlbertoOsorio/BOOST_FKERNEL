## BOOST es un script que recibe un fieldmap en mT y entrega un fieldmap en mT
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT
using GLMakie
using GeometryBasics

include("utils/grid_utils.jl")
include("utils/ppms.jl")


const FILE = "data/By_SH.jld2"                      # Ajusta si cambiaste el nombre
@load FILE By_grid xg yg zg modelBy x y z By   # Todo en mT y mm
fieldmap = By_grid

# Definir el tamaño del cascaron en el que mediremos los errores
Rmin = 0.00   # mm
Rmax = 100.0  # mm

# Definir los anillos en las bandejas en los que pondremos imanes para hacer shimming mas los que están usados
positions_in_tray_occupied   = Int[]                 # los que ya están
#positions_in_tray_new_wished = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3 ,-2, -1 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
positions_in_tray_new_wished = vcat(-21:-1, 1:21)
title_1 = "posiciones_imanes_shimming"               # nombre de la figura

#Definimos el step de la grilla del fieldmap. Viene del jld2
dx = length(xg) > 1 ? minimum(abs.(diff(xg))) : 0.0
dy = length(yg) > 1 ? minimum(abs.(diff(yg))) : 0.0
dz = length(zg) > 1 ? minimum(abs.(diff(zg))) : 0.0
resmm = (dx, dy, dz)  

cx, cy, cz = modelBy.center

# radios en cada voxel (CPU)
Rx = reshape(xg .- cx, :, 1, 1)
Ry = reshape(yg .- cy, 1, :, 1)
Rz = reshape(zg .- cz, 1, 1, :)
rgrid = sqrt.(Rx.^2 .+ Ry.^2 .+ Rz.^2)

# Definimos dmask
Δ  = max(dx, dy, dz) 
tol  = 1e-3 * Δ 
mask_shell_bool = (rgrid .>= (Rmin - tol)) .& (rgrid .<= (Rmax + tol))
dmask    = Float32.(mask_shell_bool)                

# resoluciones en metros
dx_m = Float32(dx * 1e-3)
dy_m = Float32(dy * 1e-3)
dz_m = Float32(dz * 1e-3)
dims = size(fieldmap)

posiciones = positions_from_rings_mm(positions_in_tray_new_wished;
    occupied_trays         = positions_in_tray_occupied,
    shim_radius_mm         = 235.0,
    mags_per_segment       = 7,
    num_segments           = 12,
    angle_per_segment_deg  = 2*(180 - 169.68),
    angular_offset_deg     = 0.0)

pos = posiciones
# Convertimos el Vector{NTuple{3,Float64}} a tres vectores x,y,z
xs = [p[1] for p in pos]
ys = [p[2] for p in pos]
zs = [p[3] for p in pos]

# 2) Definimos el cilindro interior (diámetro 200 mm → radio 100 mm)
R_inner = 200.0  # mm

zmin = minimum(zs)
zmax = maximum(zs)

# Opcional: un pequeño margen para que el cilindro sobresalga un poco
margin = 10.0
z0 = zmin - margin
z1 = zmax + margin

inner_cyl = Cylinder(Point3f(0, 0, z0), Point3f(0, 0, z1), R_inner)

# 3) Figura y ejes 3D
fig = Figure(resolution = (900, 900))
ax = Axis3(fig[1, 1],
    xlabel = "x [mm]",
    ylabel = "y [mm]",
    zlabel = "z [mm]",
    title  = "Imanes + cilindro interior 200 mm diámetro",
    aspect = :data,   # misma escala en x,y,z
)

# 4) Graficar el cilindro interior
mesh!(ax, inner_cyl;
    color = (:dodgerblue, 0.3),  # color + transparencia
    shading = false,
)

# 5) Graficar los puntos de los imanes
scatter!(ax, xs, ys, zs;
    markersize = 6,
    color      = :red,
)


fig
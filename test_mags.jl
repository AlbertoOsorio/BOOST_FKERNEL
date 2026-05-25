### BOOST es un script que recibe un fieldmap en mT y entrega un fieldmap en mT
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra
using Evolutionary, Random, CUDA
using DelimitedFiles, MAT
using GLMakie

include("utils/grid_utils.jl")
include("utils/pos_trays.jl")

# Ahora defino constantes
const B1CM_T = 0.012     # campo de cada iman a 1cm
const DISC_5 = true      # Las rotaciones de cada iman solo pueden ser un cm
#const λ  = 0.5           # peso RMS(∂B/∂*) en mT/m (solo cascarón)en nuestra funcion objetivo este es el λ
const w = 1.0            # peso rango (max-min)/mean en mT (solo cascarón) en nuestra funcion objetivo esto es 1

mode = "RMS"            # RMS o STDIV
const BATCH_M = 64


const FILE = "data/one_radius/By_SH_oneradius_JOSH.jld2" 
@load FILE By_grid xg yg zg 
fieldmap = By_grid

# Definir el tamaño del cascaron en el que mediremos los errores
Rmin = 0.00   # mm
Rmax = 100.0 # mm


positions_in_tray_occupied   = Int[]                
positions_in_tray_new_wished = vcat(-6:-1, 1:6)    

#Definimos el step de la grilla del fieldmap. Viene del jld2
dx = length(xg) > 1 ? minimum(abs.(diff(xg))) : 0.0
dy = length(yg) > 1 ? minimum(abs.(diff(yg))) : 0.0
dz = length(zg) > 1 ? minimum(abs.(diff(zg))) : 0.0
resmm = (dx, dy, dz)  

cx, cy, cz = 0.0, 0.0, 0.0

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
    shim_radius_mm         = 275.0,
    mags_per_segment       = 7,
    num_segments           = 12,
    angle_per_segment_deg  = 2*(180 - 169.68),
    angular_offset_deg     = 0.0)
Nmagshim = length(posiciones)

lower = fill(0.0,   Nmagshim)                  # grados
upper = fill(360.0, Nmagshim)
θ0    = 150.0 .* ones(Nmagshim)
μ_base = 0.06 .* ones(Nmagshim)                # TODO Determinar valor real de la magnitud de los imanes de shimming

P_cpu = hcat(posiciones...)             # Convierte a matrix 3x336





function save_to_csv(Psave, file, name)
    @load file λ bestθ final_state ppm
    A = Psave'
    z = A[:, 3]
    ids = cumsum([0; z[2:end] .!= z[1:end-1]])
    A[:, 3] = ids

    data = hcat(A, bestθ)
    df = DataFrame(data, [:x, :y, :z, :val])

    CSV.write("data/insert_formatted/$name.csv", df, header= ["X (mm)", "Y(mm)", "RingNumber", "Angle (deg)"])

end
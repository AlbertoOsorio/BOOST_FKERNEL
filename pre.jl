using CSV
using DataFrames, StaticArrays, JLD2, Statistics, LinearAlgebra

# Cargar archivo
filename = "Filtered_Parallelepiped_150x150x150_15.csv"
df = CSV.read(filename, DataFrame, header=10)

rename!(df, :NombreOriginal => :By)

# Extraer ejes
xg = sort(unique(df.x))
yg = sort(unique(df.y))
zg = sort(unique(df.z))

# Construir array
sort!(df, [:z, :y, :x])
nx, ny, nz = length(xg), length(yg), length(zg)
By_3D = reshape(df.By, (nx, ny, nz))

save_FILE = "data/parallelepiped_150x150x150.jld2"
@save save_FILE  By_3D xg yg zg

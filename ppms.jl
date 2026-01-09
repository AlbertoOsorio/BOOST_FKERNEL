using Statistics
function calcular_ppm(campo)
    max=maximum(campo[:])
    min=minimum(campo[:])
    p = mean(campo[:])
    ppm=1000000*(max-min)/p
    return ppm
end

############ Rings/Trays helpers (mm coherentes con tu pipeline) ############

# TrayNr -> posición axial (mm). Slots cada 10 mm; desfase ±5 mm según signo.
# Ej.: tray=+1 -> +5 mm; tray=-1 -> -5 mm; tray=+2 -> +15 mm; etc.
ringpos_from_tray_mm(trays::AbstractVector{<:Integer};
                     slot_spacing_mm::Real = 10.0,
                     half_shift_mm::Real   = 5.0) :: Vector{Float64} =
    [t > 0 ? slot_spacing_mm*t - half_shift_mm :
             slot_spacing_mm*t + half_shift_mm  for t in trays] .|> Float64

# Filtra posiciones deseadas sacando las ya ocupadas
function filter_free_trays(wished::AbstractVector{<:Integer},
                           occupied::AbstractVector{<:Integer}=Int[])::Vector{Int}
    free = Int[]
    occ = Set(occupied)
    for p in wished
        if p in occ
            @info "Tray $p ocupado: se omite"
        else
            push!(free, p)
        end
    end
    return free
end

"""
positions_from_rings_mm(wished_trays; occupied_trays=[], shim_radius_mm=235.0,
                        mags_per_segment=7, num_segments=12, angle_per_segment_deg=2*(180-169.68),
                        angular_offset_deg=0.0)

Devuelve un Vector{NTuple{3,Float64}} con posiciones (mm) de TODOS los imanes
según anillos/trays y tu geometría. Aplica la transformación de ejes:
X → -X ; Z → Y ; Y → Z
"""
function positions_from_rings_mm(wished_trays::AbstractVector{<:Integer};
                                 occupied_trays::AbstractVector{<:Integer}=Int[],
                                 shim_radius_mm::Real = 235.0,
                                 mags_per_segment::Integer = 7,
                                 num_segments::Integer = 12,
                                 angle_per_segment_deg::Real = 2*(180 - 169.68),
                                 angular_offset_deg::Real = 0.0)

    # 1) trays válidos y posiciones axiales (mm)
    free_trays = filter_free_trays(wished_trays, occupied_trays)
    ring_z_mm  = ringpos_from_tray_mm(free_trays)  # mm

    # 2) ángulos por segmento y por imán (grados)
    segment_angles_deg = collect(range(0, stop=360, length=num_segments+1))[1:end-1]
    mag_angles_deg     = collect(range(-angle_per_segment_deg/2,
                                       stop=+angle_per_segment_deg/2,
                                       length=mags_per_segment))
    offset_deg = angular_offset_deg

    # 3) construye posiciones en mm, aplica tu transformación de ejes
    pos = []
    pos_size = length(ring_z_mm) * num_segments * mags_per_segment
    sizehint!(pos, pos_size)

    for z_mm in ring_z_mm
        for seg_deg in segment_angles_deg
            θ_seg = seg_deg + offset_deg
            for mag_deg in mag_angles_deg
                θ = deg2rad(θ_seg + mag_deg)
                x_mm = shim_radius_mm * cos(θ)
                y_mm = z_mm                         # axial (a lo largo del bore)
                z_mm_circ = shim_radius_mm * sin(θ)

                # Transformación: X → -X ; Z → Y ; Y → Z
                # original (x, y, z)  ->  (-x, z, y)
                pos_trans = [-x_mm, z_mm_circ, y_mm]
                push!(pos, pos_trans)
            end
        end
    end

    @info "Generadas $(length(ring_z_mm)) rings, $(length(pos)) posiciones."
    return pos
end

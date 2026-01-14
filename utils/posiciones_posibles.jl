# posiciones.jl
# Genera posiciones cartesianas (mm) sobre el contorno de un cilindro,
# con stacking radial (hasta 3 imanes por sitio).

# Helpers
deg2rad(φ) = φ * (π/180)

"""
    generate_cylindrical_positions(;
        radius_mm::Real = 400.0,
        n_az::Int = 40,
        n_z::Int = 20,
        z_min_mm::Real = -200.0,
        z_max_mm::Real = 200.0,
        center_mm::NTuple{3,Real} = (0.0, 0.0, 0.0),
        n_radial::Int = 1,
        radial_pitch_mm::Real = 20.0
    ) -> (sites_base::Matrix{Float64}, sites_all::Matrix{Float64})

Crea una malla de posiciones sobre la **superficie lateral** de un cilindro de radio `radius_mm`.
- Total de sitios base = `n_az * n_z`.
- Cada sitio puede tener hasta `n_radial` imanes alineados **radialmente** con separación `radial_pitch_mm`.
- Devuelve:
  - `sites_base`: matriz (N_base × 3) con puntos sobre la superficie (sin stacking).
  - `sites_all` : matriz (N_base * n_radial × 3) con los shifts radiales aplicados.

Notas:
- El eje del cilindro es el **eje Z**.
- Las normales radiales apuntan en el plano XY: `u_r = (cosφ, sinφ, 0)`.
- Si `n_radial == 1`, `sites_all == sites_base`.
"""
function generate_cylindrical_positions(;
    radius_mm::Real,
    n_az::Int ,
    n_z::Int ,
    z_min_mm::Real,
    z_max_mm::Real,
    center_mm::NTuple{3,Real} = (0.0, 0.0, 0.0),
    n_radial::Int = 1,
    radial_pitch_mm::Real = 20.0
)
    @assert n_az > 0 && n_z > 0 "n_az y n_z deben ser positivos."
    @assert n_radial ≥ 1 && n_radial ≤ 3 "n_radial debe estar entre 1 y 3."
    @assert radius_mm > 0 "El radio debe ser positivo."

    cx, cy, cz = center_mm

    # Ángulos azimutales (sin repetir el 2π)
    phis = range(0, 2π; length = n_az + 1) |> collect
    pop!(phis)  # remover el último (2π) para evitar duplicar el primer punto

    # Niveles en Z
    zs = range(z_min_mm, z_max_mm; length = n_z)

    # Sitios base en la superficie del cilindro (N_base × 3)
    sites_base = Matrix{Float64}(undef, n_az * n_z, 3)
    idx = 1
    for z in zs, φ in phis
        x = cx + radius_mm * cos(φ)
        y = cy + radius_mm * sin(φ)
        sites_base[idx, 1] = x
        sites_base[idx, 2] = y
        sites_base[idx, 3] = cz + z
        idx += 1
    end

    # Stacking radial (n_radial posiciones a lo largo de u_r por sitio)
    if n_radial == 1
        # Sin stacking: sites_all = sites_base
        return sites_base, copy(sites_base)
    end

    # Offsets centrados: por ejemplo, n_radial=3 → [-pitch, 0, +pitch]
    offsets = [ (i - (n_radial + 1)/2) * radial_pitch_mm for i in 1:n_radial ]

    sites_all = Matrix{Float64}(undef, size(sites_base, 1) * n_radial, 3)
    idx_all = 1
    # Para conocer u_r de cada sitio necesitamos su φ; lo recomputamos coherentemente:
    idx = 1
    for z in zs, φ in phis
        # Base sobre la superficie
        xb = cx + radius_mm * cos(φ)
        yb = cy + radius_mm * sin(φ)
        zb = cz + z
        # Vector radial unitario en el plano XY
        urx, ury = cos(φ), sin(φ)

        for δ in offsets
            # Desplazamiento radial δ en mm (positivo = hacia afuera, negativo = hacia el centro)
            x = xb + δ * urx
            y = yb + δ * ury
            sites_all[idx_all, 1] = x
            sites_all[idx_all, 2] = y
            sites_all[idx_all, 3] = zb
            idx_all += 1
        end
        idx += 1
    end

    return sites_base, sites_all
end

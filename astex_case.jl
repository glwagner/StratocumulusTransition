# ASTEX First Lagrangian stratocumulus-to-cumulus transition: case data.
#
# Loads the authoritative EUCLIPSE/GASS case file `astex_input_v5.nc`
# (van der Dussen et al. 2013, JAMES, doi:10.1002/jame.20033) and exposes:
#
#   * z-interpolators for the initial profiles θ_l(z), q_t(z), u(z), v(z), and the
#     ozone volume mixing ratio O₃(z) (converted from the file's mass mixing ratio),
#   * time-interpolators (argument in seconds) for the time-varying boundary
#     conditions: SST(t), large-scale divergence D(t), geostrophic wind ug(t)/vg(t),
#     and free-atmosphere wind ufa(t)/vfa(t),
#   * the recommended LES vertical cell interfaces (from the file's half levels),
#   * the scalar case constants.
#
# The file lives in ./case_data/astex_input_v5.nc and was downloaded from
# euclipse.nl/wp3/ASTEX_Lagrangian/Files/astex_input_v5.nc.

module ASTEXCase

export load_astex_data, stretched_faces

using NCDatasets

const CASE_FILE = joinpath(@__DIR__, "case_data", "astex_input_v5.nc")

# Molar masses (g/mol) for the ozone mass-mixing-ratio → volume-mixing-ratio conversion
const MOLAR_MASS_DRY_AIR = 28.97
const MOLAR_MASS_OZONE   = 47.997

"""
    linear_interp(xs, ys)

Return a callable that linearly interpolates `ys` over the strictly increasing
knots `xs`, clamping to the end values outside `[first(xs), last(xs)]`.
"""
function linear_interp(xs::AbstractVector, ys::AbstractVector)
    x = collect(Float64, xs)
    y = collect(Float64, ys)
    @assert issorted(x) "interpolation knots must be sorted"
    return function (q)
        q ≤ x[1]   && return y[1]
        q ≥ x[end] && return y[end]
        i = searchsortedlast(x, q)
        t = (q - x[i]) / (x[i+1] - x[i])
        return y[i] + t * (y[i+1] - y[i])
    end
end

"""
    stretched_faces(segments)

Build a vector of vertical cell interfaces (faces) starting at 0 by appending
uniform segments. `segments` is a vector of `(top, Δz)` pairs, applied in order:
each segment adds faces spaced `Δz` apart up to `top`.

Used for the coarse development grid; the production grid uses the file's own grid.
"""
function stretched_faces(segments)
    faces = Float64[0.0]
    for (top, Δz) in segments
        while faces[end] < top - 1e-6
            push!(faces, faces[end] + Δz)
        end
    end
    return faces
end

strip_missing(a) = collect(Float64, coalesce.(a, NaN))

"""
    load_astex_data(; path=CASE_FILE)

Read the ASTEX case file and return a `NamedTuple` of profile/time interpolators,
the production-grid cell interfaces, and the case constants.
"""
function load_astex_data(; path = CASE_FILE)
    ds = NCDataset(path)

    z   = strip_missing(ds["height"][:])   # m
    t   = strip_missing(ds["tsec"][:])      # s since start

    # Initial vertical profiles (functions of height z in meters)
    θl   = linear_interp(z, strip_missing(ds["thetal"][:]))   # K
    qt   = linear_interp(z, strip_missing(ds["qt"][:]))       # kg/kg
    u    = linear_interp(z, strip_missing(ds["u"][:]))        # m/s
    v    = linear_interp(z, strip_missing(ds["v"][:]))        # m/s

    # Ozone: file stores mass mixing ratio (kg/kg); RRTMGP background gases are
    # volume mixing ratios, so vmr = mmr × (M_air / M_O₃).
    o3_mmr = strip_missing(ds["o3mmr"][:])
    o3_vmr_values = o3_mmr .* (MOLAR_MASS_DRY_AIR / MOLAR_MASS_OZONE)
    O₃ = linear_interp(z, o3_vmr_values)

    # Time-varying boundary conditions (functions of time in seconds)
    SST = linear_interp(t, strip_missing(ds["Tg"][:]))    # K
    D   = linear_interp(t, strip_missing(ds["div"][:]))   # 1/s
    ug  = linear_interp(t, strip_missing(ds["ug"][:]))    # m/s
    vg  = linear_interp(t, strip_missing(ds["vg"][:]))    # m/s
    ufa = linear_interp(t, strip_missing(ds["ufa"][:]))   # m/s (free-atmosphere)
    vfa = linear_interp(t, strip_missing(ds["vfa"][:]))   # m/s (free-atmosphere)

    # Recommended LES grid: the file's half levels ("zh") are the cell interfaces.
    z_faces = strip_missing(ds["zh"][:])
    z_faces[1] = 0.0  # ensure the bottom face sits exactly at the surface

    close(ds)

    constants = (
        latitude               = 34.0,      # °N
        longitude              = -25.0,     # °E (25°W)
        surface_pressure       = 102900.0,  # Pa
        reference_pressure     = 1.0e5,     # Pa (θ reference)
        droplet_concentration  = 100.0e6,   # m⁻³ (100 cm⁻³)
        inversion_height       = 662.5,     # m (initial)
        solar_constant         = 1376.0,    # W/m²
        duration               = 40.0 * 3600.0,  # s
        start_year             = 1992,
        start_month            = 6,
        start_day              = 13,
    )

    return (; θl, qt, u, v, O₃, SST, D, ug, vg, ufa, vfa,
              production_z_faces = z_faces,
              constants)
end

end # module

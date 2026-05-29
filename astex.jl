# # ASTEX stratocumulus-to-cumulus transition (van der Dussen et al. 2013)
#
# Large-eddy simulation of the GASS/EUCLIPSE ASTEX First Lagrangian case
# (van der Dussen et al. 2013, JAMES, doi:10.1002/jame.20033): a 40-hour Lagrangian
# transition from a well-mixed stratocumulus-topped boundary layer to a regime where
# shallow cumuli penetrate a thin, broken stratocumulus deck. The transition is driven
# by a warming sea surface, weakening large-scale subsidence, and — crucially —
# interactive radiation (cloud-top longwave cooling).
#
# All initial profiles and time-varying forcings come from the authoritative EUCLIPSE
# case file (see `astex_case.jl`). Run with
#
#     ASTEX_CONFIG=dev        julia --project astex.jl    # coarse, ~2 h, CPU-friendly
#     ASTEX_CONFIG=production julia --project astex.jl    # paper config, 40 h, GPU
#
# (`dev` is the default.)

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!

using CloudMicrophysics   # loads the BreezeCloudMicrophysicsExt extension
using RRTMGP              # loads the BreezeRRTMGPExt extension
using NCDatasets          # RRTMGP lookup tables + case file
using CUDA
using Printf
using Random
using Dates

include("astex_case.jl")
using .ASTEXCase

Random.seed!(2718)
if CUDA.functional()
    CUDA.seed!(2718)
end

# ## Configuration

config = Symbol(get(ENV, "ASTEX_CONFIG", "dev"))
@info "ASTEX configuration: $config"

Oceananigans.defaults.FloatType = Float32

data = load_astex_data()
c = data.constants

if config === :production
    arch = CUDA.functional() ? GPU() : CPU()
    Nx = Ny = 128
    Lx = Ly = 4480.0                       # m  (Δx = Δy = 35 m)
    z_faces = data.production_z_faces        # file grid: Δz ≈ 5 m through the inversion
    stop_time = c.duration                   # 40 h
    radiation_interval = 5                   # radiation every 5 iterations
    Δt₀ = 1.0
else # :dev
    arch = CUDA.functional() ? GPU() : CPU()
    Nx = Ny = 32
    Lx = Ly = 4480.0                         # m  (Δx = Δy = 140 m, coarse)
    z_faces = stretched_faces([(1600.0, 40.0), (3000.0, 80.0)])
    stop_time = 2hours
    radiation_interval = 20                  # radiation less often to keep dev cheap
    Δt₀ = 1.0
end

# Optional stop-time override (hours), e.g. ASTEX_STOP=0.02 for a quick build check.
haskey(ENV, "ASTEX_STOP") && (stop_time = parse(Float64, ENV["ASTEX_STOP"]) * hours)

Nz = length(z_faces) - 1
ztop = z_faces[end]
@info "Grid: $Nx × $Ny × $Nz, domain $(Lx) × $(Ly) × $(round(ztop)) m on $(arch); stop_time = $(stop_time)"

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and anelastic dynamics
#
# Surface pressure pₛ = 1029 hPa, reference θ ≈ surface θ_l = 288 K (van der Dussen 2013).

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants;
                                 surface_pressure = c.surface_pressure,
                                 potential_temperature = 288.0)
dynamics = AnelasticDynamics(reference_state)

# ## Microphysics: warm 1-moment with prognostic rain

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

weno = WENO(order=5)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))
momentum_advection = weno
scalar_advection = (ρθ  = weno,
                    ρqᵉ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno)

# ## Sea surface temperature (shared by surface fluxes and radiation)
#
# The SST is horizontally uniform but increases in time (Lagrangian equatorward drift).
# We store it in a 2D surface `Field` that the hourly callback updates, and we pass the
# same field to both the bulk surface fluxes and the radiation model.

SST = Field{Center, Center, Nothing}(grid)
set!(SST, data.SST(0.0))  # horizontally uniform; updated hourly from the time series

# ## Interactive radiation (RRTMGP all-sky)
#
# Full-spectrum radiation with cloud optics. The droplet effective radius is diagnosed
# from the prescribed cloud-droplet number concentration Nc = 100 cm⁻³ (van der Dussen
# 2013), and the ozone profile comes from the case file. Solar position is computed from
# the model clock and the ASTEX location (34°N, 25°W).

clock = Clock(time = DateTime(c.start_year, c.start_month, c.start_day, 0, 0, 0))

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_temperature = SST,
                                   surface_emissivity = 0.98,
                                   surface_albedo = 0.07,
                                   solar_constant = c.solar_constant,
                                   solar_position = ApparentSolarPosition(coordinate=(c.longitude, c.latitude)),
                                   background_atmosphere = BackgroundAtmosphere(O₃ = data.O₃),
                                   liquid_effective_radius = DropletNumberConcentrationRadius(c.droplet_concentration),
                                   schedule = IterationInterval(radiation_interval))

# ## Surface fluxes from the SST (bulk aerodynamic formulae)

filtered_velocities = FilteredSurfaceVelocities(grid; filter_timescale=10minutes)
coefficient = PolynomialCoefficient(roughness_length = 1.5e-4)

# The drag direction is inferred from the boundary-condition key (ρu → x, ρv → y),
# so a single BulkDrag serves both momentum components (as in the SST example).
ρuv_drag = BulkDrag(coefficient=coefficient; surface_temperature=SST, filtered_velocities)
ρe_flux  = BulkSensibleHeatFlux(coefficient=coefficient; surface_temperature=SST, filtered_velocities)
ρqᵉ_flux = BulkVaporFlux(coefficient=coefficient; surface_temperature=SST, filtered_velocities)

boundary_conditions = (ρu  = FieldBoundaryConditions(bottom=ρuv_drag),
                       ρv  = FieldBoundaryConditions(bottom=ρuv_drag),
                       ρe  = FieldBoundaryConditions(bottom=ρe_flux),
                       ρqᵉ = FieldBoundaryConditions(bottom=ρqᵉ_flux))

# ## Coriolis

coriolis = FPlane(latitude = c.latitude)

# ## Large-scale forcings (time-varying)
#
# Subsidence wˢ(z,t) = −D(t)·min(z, 1600 m) (constant above 1600 m), geostrophic wind
# ug(t)/vg(t), and a top sponge that damps w → 0 and nudges u,v toward the observed
# free-atmosphere wind ufa(t)/vfa(t). The subsidence-velocity and geostrophic-velocity
# `Field`s are mutated hourly by `update_forcings!`; the sponge targets read the clock
# time directly (Oceananigans `Relaxation` targets are functions of x,y,z,t).

D_LIMIT_HEIGHT = 1600.0

wˢ = Field{Nothing, Nothing, Face}(grid)
set!(wˢ, z -> -data.D(0.0) * min(z, D_LIMIT_HEIGHT))
subsidence = SubsidenceForcing(wˢ)

ug = Field{Nothing, Nothing, Center}(grid)
vg = Field{Nothing, Nothing, Center}(grid)
set!(ug, z -> data.ug(0.0))
set!(vg, z -> data.vg(0.0))
geostrophic = geostrophic_forcings(ug, vg)

sponge_rate = 1 / 60                      # s⁻¹ (1-minute damping timescale)
sponge_mask = GaussianMask{:z}(center=ztop, width=200)
u_sponge = Relaxation(rate=sponge_rate, mask=sponge_mask, target=(x, y, z, t) -> data.ufa(t))
v_sponge = Relaxation(rate=sponge_rate, mask=sponge_mask, target=(x, y, z, t) -> data.vfa(t))
w_sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

forcing = (u  = (subsidence, geostrophic.u, u_sponge),
           v  = (subsidence, geostrophic.v, v_sponge),
           w  = w_sponge,
           θ  = subsidence,
           qᵉ = subsidence)

# ## Model

model = AtmosphereModel(grid; clock, dynamics, coriolis, microphysics, radiation,
                        momentum_advection, scalar_advection, forcing, boundary_conditions)

# ## Initial conditions
#
# Mixed-layer θ_l = 288 K, q_t = 10.2 g/kg below the inversion (≈662 m), with the
# free-tropospheric profiles above, all read from the case file. Random perturbations
# (θ_l ∈ ±0.1 K, q_t ∈ ±2.5×10⁻⁵ kg/kg) are applied below the inversion to seed turbulence.

zi = c.inversion_height
δθ = 0.1       # K
δq = 2.5e-5    # kg/kg

ϵ() = rand() - 0.5
θᵢ(x, y, z) = data.θl(z) + 2δθ * ϵ() * (z < zi)
qᵢ(x, y, z) = data.qt(z) + 2δq * ϵ() * (z < zi)
uᵢ(x, y, z) = data.u(z)
vᵢ(x, y, z) = data.v(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ, v=vᵢ)

# ## Time-varying forcing callback
#
# Each hour, refresh the SST, subsidence velocity, and geostrophic wind from the case
# time series at the current model time.

function update_forcings!(sim)
    t = sim.model.clock.time
    set!(SST, data.SST(t))
    set!(wˢ, z -> -data.D(t) * min(z, D_LIMIT_HEIGHT))
    set!(ug, z -> data.ug(t))
    set!(vg, z -> data.vg(t))
    fill_halo_regions!(SST)
    fill_halo_regions!(wˢ)
    fill_halo_regions!(ug)
    fill_halo_regions!(vg)
    return nothing
end

# ## Simulation

simulation = Simulation(model; Δt=Δt₀, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=5.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

add_callback!(simulation, update_forcings!, TimeInterval(1hour))

# ## Diagnostics

θl  = liquid_ice_potential_temperature(model)
qˡ  = model.microphysical_fields.qˡ    # total liquid (cloud + rain)
qᶜˡ = model.microphysical_fields.qᶜˡ   # cloud liquid only
qʳ  = model.microphysical_fields.qʳ    # rain mass fraction
qᵛ  = model.microphysical_fields.qᵛ
ρʳ  = model.dynamics.reference_state.density

# Cloud liquid water path (column-integrated cloud liquid), kg/m²
LWP = Field(Integral(ρʳ * qᶜˡ, dims=3))

function progress(sim)
    t = sim.model.clock.time
    wmax = maximum(abs, sim.model.velocities.w)
    qˡmax = maximum(qᶜˡ)
    compute!(LWP)
    lwp = maximum(LWP)
    msg = @sprintf("iter %d, t %s, Δt %s | SST %.2f K, D %.1e | max|w| %.2e, max qˡ %.2e, max LWP %.1f g/m²",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   data.SST(t), data.D(t), wmax, qˡmax, 1000 * lwp)
    @info msg
    return nothing
end
add_callback!(simulation, progress, IterationInterval(100))

# ## Output

prefix = "astex_$(config)"

qᵗ = qˡ + qᵛ
w² = model.velocities.w^2
profile_outputs = (; θl, qᵛ, qˡ, qᶜˡ, qʳ, qᵗ, u=model.velocities.u, v=model.velocities.v, w²)
avg_outputs = NamedTuple(name => Average(profile_outputs[name], dims=(1, 2)) for name in keys(profile_outputs))
avg_outputs = merge(avg_outputs, (; LWP = Average(LWP, dims=(1, 2))))

simulation.output_writers[:profiles] = JLD2Writer(model, avg_outputs;
    filename = "$(prefix)_profiles.jld2",
    schedule = AveragedTimeInterval(30minutes),
    overwrite_existing = true)

# Find a near-cloud-base level for the horizontal slices
zc = znodes(grid, Center())
k = searchsortedfirst(zc, min(600.0, ztop / 2))
@info "Saving x–y slices at z = $(round(zc[k])) m (k = $k)"

w = model.velocities.w
slice_outputs = (wxz = view(w, :, 1, :), qᶜˡxz = view(qᶜˡ, :, 1, :),
                 wxy = view(w, :, :, k), qᶜˡxy = view(qᶜˡ, :, :, k),
                 LWP = LWP)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
    filename = "$(prefix)_slices.jld2",
    schedule = TimeInterval(5minutes),
    overwrite_existing = true)

@info "Running ASTEX ($config) simulation..."
run!(simulation)
@info "Done. Profiles → $(prefix)_profiles.jld2, slices → $(prefix)_slices.jld2"

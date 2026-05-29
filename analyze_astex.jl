# # ASTEX transition diagnostics
#
# Post-processing for the ASTEX stratocumulus-to-cumulus transition simulation
# (van der Dussen et al. 2013). Produces:
#   1. hourly horizontally-averaged profile evolution (θ_l, q_t, cloud liquid, w²),
#   2. time series of liquid water path, cloud cover, and inversion height,
#   3. an animation of vertical-velocity and cloud-liquid slices.
#
#     ASTEX_CONFIG=dev julia --project analyze_astex.jl

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics

config = Symbol(get(ENV, "ASTEX_CONFIG", "dev"))
prefix = "astex_$(config)"
profiles_file = "$(prefix)_profiles.jld2"
slices_file   = "$(prefix)_slices.jld2"

@info "Analyzing $profiles_file and $slices_file"

# ## Profile evolution

θlt  = FieldTimeSeries(profiles_file, "θl")
qᵗt  = FieldTimeSeries(profiles_file, "qᵗ")
qᶜˡt = FieldTimeSeries(profiles_file, "qᶜˡ")
w²t  = FieldTimeSeries(profiles_file, "w²")
lwpt = FieldTimeSeries(profiles_file, "LWP")

z = znodes(θlt.grid, Center())
times = θlt.times
Nt = length(times)

fig = Figure(size=(1100, 800), fontsize=14)
axθ  = Axis(fig[1, 1], xlabel="θ_l (K)",      ylabel="z (m)")
axq  = Axis(fig[1, 2], xlabel="q_t (g/kg)",   ylabel="z (m)")
axqc = Axis(fig[2, 1], xlabel="q_cl (g/kg)",  ylabel="z (m)")
axw  = Axis(fig[2, 2], xlabel="⟨w²⟩ (m²/s²)", ylabel="z (m)")

colors = cgrad(:viridis, max(Nt, 2); categorical=true)
for n in 1:Nt
    label = @sprintf("%.0f h", times[n] / 3600)
    lines!(axθ,  interior(θlt[n], 1, 1, :),        z, color=colors[n], label=label)
    lines!(axq,  1e3 .* interior(qᵗt[n], 1, 1, :), z, color=colors[n])
    lines!(axqc, 1e3 .* interior(qᶜˡt[n], 1, 1, :), z, color=colors[n])
    lines!(axw,  interior(w²t[n], 1, 1, :),        z, color=colors[n])
end
Legend(fig[1:2, 3], axθ, "time", framevisible=false)
Label(fig[0, :], "ASTEX transition: mean profile evolution ($config)", fontsize=18, tellwidth=false)
save("$(prefix)_profiles.png", fig)
@info "Wrote $(prefix)_profiles.png"

# ## Time series: LWP, cloud cover, inversion height
#
# Cloud cover is the fraction of columns whose liquid water path exceeds 20 g/m²
# (a common stratocumulus threshold), computed from the saved 2D LWP slices.
# Inversion height is diagnosed as the level of strongest θ_l gradient.

LWPt = FieldTimeSeries(slices_file, "LWP")
st = LWPt.times
threshold = 20e-3  # kg/m²
cloud_cover = [mean(interior(LWPt[n]) .> threshold) for n in 1:length(st)]
mean_lwp    = [mean(interior(LWPt[n])) for n in 1:length(st)]

function inversion_height(θl_profile, z)
    dθdz = diff(θl_profile) ./ diff(z)
    k = argmax(dθdz)
    return (z[k] + z[k+1]) / 2
end
zi = [inversion_height(interior(θlt[n], 1, 1, :), z) for n in 1:Nt]

figts = Figure(size=(1100, 350), fontsize=14)
axlwp = Axis(figts[1, 1], xlabel="time (h)", ylabel="LWP (g/m²)", title="Mean liquid water path")
axcc  = Axis(figts[1, 2], xlabel="time (h)", ylabel="cloud cover", title="Cloud cover (LWP > 20 g/m²)")
axzi  = Axis(figts[1, 3], xlabel="time (h)", ylabel="z_i (m)",     title="Inversion height")
lines!(axlwp, st ./ 3600, 1e3 .* mean_lwp, color=:steelblue, linewidth=2)
lines!(axcc,  st ./ 3600, cloud_cover,      color=:seagreen,  linewidth=2)
lines!(axzi,  times ./ 3600, zi,            color=:firebrick, linewidth=2)
ylims!(axcc, 0, 1.05)
save("$(prefix)_timeseries.png", figts)
@info "Wrote $(prefix)_timeseries.png"

# ## Slice animation

wxz  = FieldTimeSeries(slices_file, "wxz")
qcxz = FieldTimeSeries(slices_file, "qᶜˡxz")
wxy  = FieldTimeSeries(slices_file, "wxy")
qcxy = FieldTimeSeries(slices_file, "qᶜˡxy")

x = xnodes(wxz.grid, Center())
zc = znodes(wxz.grid, Center())
atimes = wxz.times
Na = length(atimes)

wlim = max(maximum(abs, wxz), 1e-3) / 2
qlim = max(maximum(qcxz), 1e-6)

fanim = Figure(size=(1000, 700), fontsize=14)
axwxz = Axis(fanim[2, 1], xlabel="x (m)", ylabel="z (m)", title="w (m/s)")
axqxz = Axis(fanim[2, 2], xlabel="x (m)", ylabel="z (m)", title="cloud liquid (kg/kg)")
n = Observable(1)
wxzn  = @lift interior(wxz[$n],  :, 1, :)
qcxzn = @lift interior(qcxz[$n], :, 1, :)
hmw = heatmap!(axwxz, x, zc, wxzn,  colormap=:balance, colorrange=(-wlim, wlim))
hmq = heatmap!(axqxz, x, zc, qcxzn, colormap=Reverse(:Blues_4), colorrange=(0, qlim))
Colorbar(fanim[2, 0], hmw, flipaxis=false)
Colorbar(fanim[2, 3], hmq)
title = @lift "ASTEX $config — t = " * prettytime(atimes[$n])
Label(fanim[1, :], title, fontsize=18, tellwidth=false)

record(fanim, "$(prefix)_slices.mp4", 1:Na, framerate=12) do nn
    n[] = nn
end
@info "Wrote $(prefix)_slices.mp4"

using ITensors
using ITensorMPS
using CSV, DataFrames

# --- params ---
L = 10
Nmax = 4                 # allow 0..4 bosons per site
Nsites = 2L              # ladder has 2*L sites
J, Jpar, U = 1.0, 5.0, 0.0
J_ratio = Jpar/J

sites = siteinds("Boson", Nsites; dim=Nmax+1, conserve_number=true)

site_index(j,m) = (m-1)*L + j  # map (rung j, leg m∈{1,2}) -> 1..2L

function build_H(sites; χ, J=1.0, Jpar=0.5)
    os = OpSum()
    # legs with Peierls phases
    for j in 1:L-1
        i1,i2 = site_index(j,1), site_index(j+1,1)
        k1,k2 = site_index(j,2), site_index(j+1,2)
        os += -Jpar*exp(-im*χ), "Adag", i1, "A", i2
        os += -Jpar*exp(+im*χ), "Adag", i2, "A", i1
        os += -Jpar*exp(+im*χ), "Adag", k1, "A", k2
        os += -Jpar*exp(-im*χ), "Adag", k2, "A", k1
    end
    # rungs
    for j in 1:L
        a,b = site_index(j,1), site_index(j,2)
        os += -J, "Adag", a, "A", b
        os += -J, "Adag", b, "A", a
    end
    # diagonals
    for j in 1:L-1
        a,b = site_index(j+1,1), site_index(j,2)
        os += -J, "Adag", a, "A", b
        os += -J, "Adag", b, "A", a
    end
    return MPO(os, sites)
end

function chiral_current(psi, sites; χ, Jpar)
    Jc = 0.0
    for j in 1:L-1
        # j_{j,1}^|| - j_{j,2}^||
        os1 = OpSum(); os2 = OpSum()
        i,ip1 = site_index(j,1), site_index(j+1,1)
        k,kp1 = site_index(j,2), site_index(j+1,2)
        os1 += -im*Jpar*exp(+im*χ), "Adag", i, "A", ip1
        os1 += +im*Jpar*exp(-im*χ), "Adag", ip1, "A", i
        os2 += -im*Jpar*exp(-im*χ), "Adag", k, "A", kp1
        os2 += +im*Jpar*exp(+im*χ), "Adag", kp1, "A", k
        Jc += inner(psi, MPO(os1,sites), psi) - inner(psi, MPO(os2,sites), psi)
    end
    return Jc/(2*(L-1))
end

function avg_rung_current(psi, sites; J)
    Jr = 0.0
    for j in 1:L
        os = OpSum()
        a,b = site_index(j,1), site_index(j,2)
        os += -im*J, "Adag", a, "A", b
        os += +im*J, "Adag", b, "A", a
        Jr += abs(inner(psi, MPO(os,sites), psi))
    end
    return Jr/L
end

# one-boson initial state
state = ["0" for _ in 1:Nsites]
state[site_index(div(L,2),1)] = "1"   # place the single boson
psi0 = MPS(sites, state)

@show sites[1]
@show flux(psi0)  # should be QN("N", 1)

sweeps = Sweeps(4);
nsweeps = 4;
setmaxdim!(sweeps, fill(10, nsweeps)...);
setcutoff!(sweeps, 1e-12)

chis = range(0, stop=π, length=10)
rows = Vector{NamedTuple}()

for χ in chis
    H = build_H(sites; χ, J, Jpar)
    E, psi = dmrg(H, psi0, sweeps)
    Jc = chiral_current(psi, sites; χ, Jpar)
    Jr = avg_rung_current(psi, sites; J)
    push!(rows, (chi=χ, energy=E, Jc=Jc, Jr=Jr))
end

df = DataFrame(rows)
fname = "ladder_single_particle_scan_J=$(J_ratio).csv"
CSV.write(fname, df)
println("Wrote ladder_single_particle_scan$(J_ratio).csv")

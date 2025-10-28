using ITensors
using ITensorMPS  
using CSV, DataFrames

# --- params ---
L      = 15
rho      = 0.5                      # half filling like the paper
N      = Int(round(rho*2L))         # total bosons

Nmax   = 2                        # paper used ≥4–5
J, Jpar, U = 1.0, 0.5, 2.5        # match paper examples
J_ratio = Jpar/J;
sites  = siteinds("Boson", 2L; dim=Nmax+1, conserve_number=true)

site_index(j,m) = (m-1)*L + j  # map (rung j, leg m∈{1,2}) -> 1..2L

function build_H(sites; χ, J=1.0, Jpar=0.5, U=0.0)
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
    # onsite interaction  (Boson site type supports "N(N-1)")
    if U != 0.0
        for i in 1:2L
            os += 0.5U, "N", i, "N", i   # (U/2) * N^2
            os += -0.5U, "N", i          # -(U/2) * N
        end
    end
    return MPO(os, sites)
end

function avg_rung_current(psi, sites; J)
    acc = 0.0
    # odd rung bonds (j,1) <-> (j,2), j = 1..L
    for j in 1:L
        a, b = site_index(j,1), site_index(j,2)
        os = OpSum()
        os += -1im*J, "Adag", a, "A", b
        os += +1im*J, "Adag", b, "A", a   # H.c. as b† b
        acc += abs(inner(psi, MPO(os, sites), psi))
    end
    # even rung bonds (j+1,1) <-> (j,2), j = 1..L-1
    for j in 1:(L-1)
        a2, b = site_index(j+1,1), site_index(j,2)
        os = OpSum()
        os += -1im*J, "Adag", a2, "A", b
        os += +1im*J, "Adag", b,  "A", a2
        acc += abs(inner(psi, MPO(os, sites), psi))
    end
    return acc/(2L - 1)
end


function chiral_current(psi, sites; χ, Jpar)
    Jc = 0.0
    for j in 1:L-1
        # top leg m=1: phase e^{-iχ}
        i, ip1 = site_index(j,1), site_index(j+1,1)
        os1 = OpSum()
        os1 += -1im*Jpar*exp(-1im*χ), "Adag", i,   "A", ip1
        os1 += +1im*Jpar*exp(+1im*χ), "Adag", ip1, "A", i

        # bottom leg m=2: phase e^{+iχ}
        k, kp1 = site_index(j,2), site_index(j+1,2)
        os2 = OpSum()
        os2 += -1im*Jpar*exp(+1im*χ), "Adag", k,   "A", kp1
        os2 += +1im*Jpar*exp(-1im*χ), "Adag", kp1, "A", k

        Jc += inner(psi, MPO(os1,sites), psi) - inner(psi, MPO(os2,sites), psi)
    end
    return Jc/(2*(L-1))
end

function half_filled_state()
    state = fill("0", 2L)
    for i in 1:N
        state[i] = "1"            # ensure total N; respects QN conservation
    end
    return state
end

psi0 = MPS(sites, half_filled_state())

nsweeps = 1
sweeps  = Sweeps(nsweeps)
setmaxdim!(sweeps, 1600)
setmindim!(sweeps,  1000)
setcutoff!(sweeps, 1e-5)

chis = range(0, stop=π, length=20)
rows = Vector{NamedTuple}()

function half_filled_mps(sites, N)
    state = fill("0", length(sites))
    for i in 1:N
        state[i] = "1"
    end
    return MPS(sites, state)
end

function run_scan(sites, chis; J, Jpar, U, sweeps, psi0)
    psi_ws = psi0   # local psi0
    rows = NamedTuple[]
    for χ in chis
        H = build_H(sites; χ, J, Jpar, U)
        H = cu(H)
        E, psi_ws = dmrg(H, psi_ws, sweeps)    # warm-started MPS
        Jc = chiral_current(psi_ws, sites; χ, Jpar=Jpar)
        Jr = avg_rung_current(psi_ws, sites; J=J)
        push!(rows, (chi=χ, energy=E, Jc=Jc, Jr=Jr))
    end
    return rows
end

rows = run_scan(sites, chis; J=J, Jpar=Jpar, U=U, sweeps=sweeps, psi0)

df = DataFrame(rows)
fname = "ladder_many_particle_scan_J=$(J_ratio)_L=$(L)_n=$(N).csv"
CSV.write(fname, df)
println("Wrote ladder_many_particle_scan_J=$(J_ratio)_L=$(L)_n=$(N).csv")
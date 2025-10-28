using ITensors, ITensorMPS
let
  N = 100
  sites = siteinds("S=1",N;conserve_qns=true)

  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(os,sites)

  state = [isodd(n) ? "Up" : "Dn" for n=1:N]
  psi0 = MPS(sites,state)
  @show flux(psi0)

  nsweeps = 5
  maxdim = [10,20,100,100,200]
  cutoff = [1E-10]

  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)

  return
end
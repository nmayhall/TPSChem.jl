using TPSChem.QCBase
using Printf
using TPSChem
using NPZ
using TPSChem.InCoreIntegrals
using TPSChem.RDM
using JLD2

@load  "data_cmf_TD_12.jld2"
M = 50
#@load "cmf_op_TD_with_ops.jld2"
display(clusters)
display(init_fspace)
ref_fspace = FockConfig(init_fspace)
ecore = ints.h0

cluster_bases = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3], ref_fspace, max_roots=M, verbose=1);

clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints);

TPSChem.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

nroots = 10
ci_vector = TPSChem.TPSCIstate(clusters, TPSChem.FockConfig(init_fspace), R=nroots);

# Add the lowest energy single exciton to basis
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([1,1])] = zeros(Float64,nroots)
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([2,1])] = zeros(Float64,nroots)
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([1,2])] = zeros(Float64,nroots)
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([3,1])] = zeros(Float64,nroots)
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([1,3])] = zeros(Float64,nroots)
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([4,1])] = zeros(Float64,nroots)
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([1,4])] = zeros(Float64,nroots)

# TT states ms=0
ci_vector[TPSChem.FockConfig(init_fspace)][TPSChem.ClusterConfig([2,2])] = zeros(Float64,nroots)



# Spin-flip states
fspace_0 = TPSChem.FockConfig(init_fspace)

## ba
tmp_fspace = TPSChem.replace(fspace_0, (1,2), ([4,2],[2,4]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1])] = zeros(Float64,nroots)


## ab
tmp_fspace = TPSChem.replace(fspace_0, (1,2), ([2,4],[4,2]))
TPSChem.add_fockconfig!(ci_vector, tmp_fspace)
ci_vector[tmp_fspace][TPSChem.ClusterConfig([1,1])] = zeros(Float64,nroots)


TPSChem.eye!(ci_vector)

#ci_vector = TPSChem.add_spin_focksectors(ci_vector)

eci, v = TPSChem.tps_ci_direct(ci_vector, cluster_ops, clustered_ham);

e0a, v0a = TPSChem.tpsci_ci(v, cluster_ops, clustered_ham,
                            incremental  = true,
                            thresh_cipsi = 1e-3,
                            thresh_foi   = 1e-5,
                            thresh_asci  = -1, 
                            max_mem_ci = 100.0);



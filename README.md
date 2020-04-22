# EBMS

Siegel et al. [35] presented the original energy-banding
(EB) algorithm for OpenMC, a distributed Monte Carlo (MC)
neutron-transport code. Felker et al. [20] extended the EB
idea to distributed-memory machines by distributing the
cross-section data (composed of energy bands) across multiple
nodes. Rather than the domain, particles are evenly distributed
between the nodes. During simulation, each node fetches one band of
the cross section using MPI_Get operations, tracks the movement
of its share of particles, and iterates over the number of bands.

The [EBMS miniapp](https://github.com/ANL-CESAR/EBMS) captures
the communication pattern of the distributed EB idea. It utilizes
MPI shared memory: multiple processes on a node that share a receive
buffer that is large enough to hold one band of the cross-section.
While the computation is distributed among the different processes
on the node, only one process is responsible for communication. This
work extends the communication of the EBMS miniapp to distribute the
communication workload among the processes as well. We also implement
a MPI+threads version of the miniapp with one multithreaded process
per node. The communication workload between the cores is the same
for both the MPI everywhere (+ shared memory) and the MPI+threads
versions.

Note that the codes in this repository only implement the
communication pattern, and not any actual computation.

# Versions

ebms_single_shared_mem.c
- MPI everywhere + MPI Shared Memory

ebms_multiple.c
- MPI+OpenMP version using MPI_THREAD_MULTIPLE
- Expressing no logical parallelism exposed to the MPI library

ebms_multiple_nwins.c
- MPI+OpenMP version using MPI_THREAD_MULTIPLE
- Expressing logical parallelism to the MPI library
  - The MPI_Get operations of each thread are independent. Hence, each thread uses their own window.

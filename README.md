1. wrapper.h
  - define and implement the wrapper interfaces, which serve for the hybrid MPI+MPI context-based collectives (broadcast, allgather and     allreduce).
  - In the hybrid MPI+MPI context-based collectives, only one copy of the replicated on-node data is maintained for the collectives.
2. bench_hy_xx
  - The micro-benchmarks demonstrating the usage of the above wrapper funtions (substitute the MPI standard collectives, e.g., MPI_Bcast,   MPI_Allgather and MPI_Allreduce)


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

void setup(int rank, int num_ranks, int num_nodes, int num_workers_per_node,
        int argc, char **argv, int *num_bands_ptr, size_t *band_size_ptr,
        int *final_flag);

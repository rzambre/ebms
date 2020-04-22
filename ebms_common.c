#include "ebms.h"

void setup(int rank, int num_ranks, int num_nodes, int num_workers_per_node,
        int argc, char **argv, int *num_bands_ptr, size_t *band_size_ptr, 
        int *final_flag)
{
    int num_bands;
    size_t band_size;

    (*final_flag) = 0;

    if (argc != 3) {
        if (!rank)
            printf("usage: ebms_mpi <#bands> <band-size>\n");
        (*final_flag) = 1;
        return;
    }

    num_bands = atoi(argv[1]);  /* number of bands */
    band_size = atoi(argv[2]);     /* size of each band in bytes (for now) */

    if (num_workers_per_node < num_nodes) {
        if (num_nodes % num_workers_per_node)
            MPI_Abort(MPI_COMM_WORLD, 1);   /* abort num_workers_per_node needs to divide num_nodes */
    } else {
        if (num_workers_per_node % num_nodes)
            MPI_Abort(MPI_COMM_WORLD, 1);   /* abort num_nodes needs to divide num_workers_per_node*/
    }

    if (band_size % num_nodes)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort num_nodes needs to divide band_size */
    if (band_size % num_ranks)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort num_ranks needs to divide band_size */
    if (band_size % num_workers_per_node)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort num_workers_per_node needs to divide band_size */

    (*num_bands_ptr) = num_bands;
    (*band_size_ptr) = band_size;
}

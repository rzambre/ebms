/* There is only one memory group in this kernel */

#include "ebms.h"

#define FINE_TIME 1
#define FETCH_TIME 0
#define DEBUG_INFO 0

int main(int argc, char **argv)
{
    int final_flag;
    int rank, num_ranks, num_nodes;
    int shm_rank, num_shm_ranks;
    int i;
    int band_i, num_bands;
    int get_i, num_gets;
    int start_target_node, start_target_rank, target_rank;
    int id_within_a_target_node;
    MPI_Aint disp;

    int *rank_array, *global_to_shm_rank_map;

    size_t size_of_each_get;
    size_t band_size;
    size_t band_size_per_node, band_buffer_per_shm_rank;
    size_t band_memory_per_node;

    char *bands_mem;
    char *band_buffer;
    char *get_into_addr;
    
#if FINE_TIME
    int get_counter, tot_get_count;
    double t_get_start, t_get, t_per_get;
    double t_flush_start, t_flush, t_per_flush;
    double max_t_get, min_t_get, mean_t_get;
    double max_t_flush, min_t_flush, mean_t_flush;
    double *t_get_procs, *t_flush_procs;
#elif FETCH_TIME
    int fetch_counter, tot_fetch_count;
    double t_fetch_start, t_fetch, t_per_fetch;
    double max_t_fetch, min_t_fetch, mean_t_fetch;
    double *t_fetch_procs;
#else
    double t1, t2;
#endif

    MPI_Comm shm_comm;
    MPI_Group comm_world_group, shm_comm_group;

    int is_node_leader, color;
    MPI_Comm node_leader_comm;

    MPI_Win shm_win, bands_win;
    //MPI_Win buffer_win;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);  
    
    /* Create the shared memory communicator */
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_rank(shm_comm, &shm_rank);
    MPI_Comm_size(shm_comm, &num_shm_ranks);
    
    /* Create communicator of leaders from each node */
    is_node_leader = (shm_rank == 0) ? 1 : 0;
    color = is_node_leader ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &node_leader_comm);
    if (is_node_leader) MPI_Comm_size(node_leader_comm, &num_nodes);
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, shm_comm);
#if DEBUG_INFO
    printf("Number of nodes is %d\n", num_nodes);
#endif

    setup(rank, num_ranks, num_nodes, num_shm_ranks, argc, argv,
            &num_bands, &band_size,
            &final_flag);
    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    /* Create a map between global to shared memory ranks */
    MPI_Comm_group(MPI_COMM_WORLD, &comm_world_group);
    MPI_Comm_group(shm_comm, &shm_comm_group);

    rank_array = malloc(num_ranks * sizeof(int));
    global_to_shm_rank_map = malloc(num_ranks * sizeof(int));
    for (i = 0; i < num_ranks; i++)
        rank_array[i] = i;
    MPI_Group_translate_ranks(comm_world_group, num_ranks, rank_array, shm_comm_group, global_to_shm_rank_map);
     
    /* The amount of bytes stored per band on each node */
    band_size_per_node = band_size / num_nodes;
    
    /* Allocate the memory for the bands on the rank of the node leader.
     * This memory can be accessed directly through shared memory for the ranks on this node */
    band_memory_per_node = is_node_leader ? band_size_per_node * num_bands : 0;
    MPI_Win_allocate_shared(band_memory_per_node, sizeof(char), MPI_INFO_NULL, shm_comm, &bands_mem, &shm_win);
    
    /* Prepare the node on the memory for access by ranks on other nodes through RMA. */
    MPI_Win_create(bands_mem, band_memory_per_node, sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &bands_win);

    if (num_shm_ranks < num_nodes) {
        /* Each rank on the node will get from more than 1 rank */
        num_gets = num_nodes / num_shm_ranks;
        size_of_each_get = band_size_per_node;
        id_within_a_target_node = 0;
        start_target_node = num_gets * shm_rank;
    } else {
        /* Each rank on a node will get only once. Concurrent gets
         * to the same target rank will have different offsets */
        num_gets = 1;
        size_of_each_get = band_size / num_shm_ranks;
        id_within_a_target_node = shm_rank % (num_shm_ranks / num_nodes);
        start_target_node = shm_rank / (num_shm_ranks / num_nodes);
    }
    /* The first rank on each node hosts the memory in bands_win */
    start_target_rank = start_target_node * num_shm_ranks;
#if DEBUG_INFO
    if (rank == 0) printf("Size of each get: %d\n", (int) size_of_each_get);
    if (rank == 0) printf("Num gets: %d\n", num_gets);
    printf("Rank %d: ID within a target node %d\n", rank, id_within_a_target_node);
    printf("Rank %d: Start target rank %d\n", rank, start_target_rank);
#endif

    /* Allocate the receive buffer for 1 band.
     * TODO: make it also as shared memory since the processes will distribute the
     * particle-tracking workload and may need access to any part of the band data.
     * This shared memory region is contiguous. */
    band_buffer_per_shm_rank = num_gets * size_of_each_get;
    band_buffer = calloc(sizeof(char), band_buffer_per_shm_rank);
    //MPI_Win_allocate_shared(band_buffer_per_shm_rank, sizeof(char), MPI_INFO_NULL, shm_comm, &band_buffer, &buffer_win);
 
    MPI_Win_lock_all(MPI_MODE_NOCHECK, bands_win);

    MPI_Barrier(MPI_COMM_WORLD);
#if DEBUG_INFO
    if (rank == 0) {
        printf("About to start!\n");
        fflush(stdout);
    }
#endif
#if FINE_TIME
    get_counter = 0;
    t_get = t_flush = 0;
#elif FETCH_TIME
    fetch_counter = 0;
    t_fetch = 0;
#else
    t1 = MPI_Wtime();
#endif
    
    for (band_i = 0; band_i < num_bands; band_i++) {
        disp = (band_i * band_size_per_node) + (id_within_a_target_node * size_of_each_get);
#if DEBUG_INFO
        printf("Rank %d: Disp for band %d is %d\n", rank, band_i, disp);
#endif
        /* Fetch my share of this band and store it in my buffer */
        for (get_i = 0; get_i < num_gets; get_i++) {
            get_into_addr = band_buffer + (get_i * size_of_each_get);
            target_rank = start_target_rank + (get_i * num_shm_ranks);
            if (global_to_shm_rank_map[target_rank] != MPI_UNDEFINED) {
                /* TODO: copy from shared memory */
            } else {
                /* Get from remote node */
#if DEBUG_INFO
                printf("Rank %d getting from rank %d\n", rank, target_rank);
#endif
#if FETCH_TIME
                fetch_counter++;
                t_fetch_start = MPI_Wtime();
#endif
#if FINE_TIME
                get_counter++;
                t_get_start = MPI_Wtime();
#endif
                MPI_Get(get_into_addr,
                    size_of_each_get, MPI_CHAR,
                    target_rank, disp,
                    size_of_each_get, MPI_CHAR,
                    bands_win);
#if FINE_TIME
                t_get += (MPI_Wtime() - t_get_start);
                t_flush_start = MPI_Wtime();
#endif
                MPI_Win_flush(target_rank, bands_win);
#if FINE_TIME
                t_flush += (MPI_Wtime() - t_flush_start);
#endif
#if FETCH_TIME
                t_fetch += (MPI_Wtime() - t_fetch_start);
#endif
            }
        }
        //MPI_Win_sync(buffer_win);
        /* Perform computation cooperatively with other shm ranks using the fetched band data */    
        MPI_Barrier(shm_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if FINE_TIME
    if (get_counter > 0) {
        t_per_get = t_get / get_counter;
        t_per_flush = t_flush / get_counter;
    } else
        t_per_get = t_per_flush = 0;

    if (rank == 0) {
        t_get_procs = calloc(num_ranks, sizeof(double));
        t_flush_procs = calloc(num_ranks, sizeof(double));
        if (!t_get_procs || !t_flush_procs) {
            fprintf(stderr, "Unable to allocate memory for t_get_procs or t_flush_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        t_get_procs = NULL;
        t_flush_procs = NULL;
    }
#if DEBUG_INFO
    printf("Rank %d: t_per_get %.9f\n", rank, t_per_get);
#endif
    MPI_Gather(&t_per_get, 1, MPI_DOUBLE, t_get_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&t_per_flush, 1, MPI_DOUBLE, t_flush_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    tot_get_count = 0;
    MPI_Reduce(&get_counter, &tot_get_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    mean_t_get = mean_t_flush = 0;
    max_t_get = max_t_flush = -1;
    min_t_get = min_t_flush = 9999;

    if (rank == 0) {
        int pi;
        double sum_t_get, sum_t_flush;
        int nworkers_who_got;

        nworkers_who_got = 0;
        sum_t_get = sum_t_flush = 0;

        for (pi = 0; pi < num_ranks; pi++) {
            if (t_get_procs[pi] > 0) {
                if (!(t_flush_procs[pi] > 0))
                    printf("THIS SHOULD NEVER OCCUR!");
                
                nworkers_who_got++;
                
                if (max_t_get < t_get_procs[pi])
                    max_t_get = t_get_procs[pi];
                if (min_t_get > t_get_procs[pi])
                    min_t_get = t_get_procs[pi];
                sum_t_get += t_get_procs[pi];
                
                if (max_t_flush < t_flush_procs[pi])
                    max_t_flush = t_flush_procs[pi];
                if (min_t_flush > t_flush_procs[pi])
                    min_t_flush = t_flush_procs[pi];
                sum_t_flush += t_flush_procs[pi];
            }
        }

        mean_t_get = sum_t_get / nworkers_who_got;
        mean_t_flush = sum_t_flush / nworkers_who_got;
    }
#elif FETCH_TIME
    if (fetch_counter > 0) {
        t_per_fetch = t_fetch / fetch_counter;
    } else
        t_per_fetch = 0;

    if (rank == 0) {
        t_fetch_procs = calloc(num_ranks, sizeof(double));
        if (!t_fetch_procs) {
            fprintf(stderr, "Unable to allocate memory for t_fetch_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        t_fetch_procs = NULL;
    }
#if DEBUG_INFO
    printf("Rank %d: t_per_fetch %.9f\n", rank, t_per_fetch);
#endif
    MPI_Gather(&t_per_fetch, 1, MPI_DOUBLE, t_fetch_procs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    tot_fetch_count = 0;
    MPI_Reduce(&fetch_counter, &tot_fetch_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    mean_t_fetch = 0;
    max_t_fetch = -1;
    min_t_fetch = 9999;

    if (rank == 0) {
        int pi;
        double sum_t_fetch;
        int nworkers_who_fetched;

        nworkers_who_fetched = 0;
        sum_t_fetch = 0;

        for (pi = 0; pi < num_ranks; pi++) {
            if (t_fetch_procs[pi] > 0) {
                nworkers_who_fetched++;
                
                if (max_t_fetch < t_fetch_procs[pi])
                    max_t_fetch = t_fetch_procs[pi];
                if (min_t_fetch > t_fetch_procs[pi])
                    min_t_fetch = t_fetch_procs[pi];
                sum_t_fetch += t_fetch_procs[pi];
            }
        }

        mean_t_fetch = sum_t_fetch / nworkers_who_fetched;
    }
#else
    t2 = MPI_Wtime();
#endif

    MPI_Win_unlock_all(bands_win);

    if (rank == 0) {
#if FINE_TIME
        printf("num_bands,band_size,nworkers,min_get_time,max_get_time,mean_get_time,get_count,min_flush_time,max_flush_time,mean_flush_time\n");
        printf("%d,%d,%d,%.9f,%.9f,%.9f,%d,%.9f,%.9f,%.9f\n", num_bands, (int) band_size, num_ranks, min_t_get, max_t_get, mean_t_get, tot_get_count, min_t_flush, max_t_flush, mean_t_flush);
        free(t_get_procs);
        free(t_flush_procs);
#elif FETCH_TIME
        printf("num_bands,band_size,nworkers,min_fetch_time,max_fetch_time,mean_fetch_time,fetch_count\n");
        printf("%d,%d,%d,%.9f,%.9f,%.9f,%d\n", num_bands, (int) band_size, num_ranks, min_t_fetch, max_t_fetch, mean_t_fetch, tot_fetch_count);
        free(t_fetch_procs);
#else
        printf("num_bands,band_size,nworkers,time\n");
        printf("%d,%d,%d,%.9f\n", num_bands, (int) band_size, num_ranks, t2-t1);
#endif
    }

    MPI_Win_free(&shm_win);
    MPI_Win_free(&bands_win);

    free(rank_array);
    free(global_to_shm_rank_map);
    free(band_buffer);

    MPI_Finalize();
    return 0;
}

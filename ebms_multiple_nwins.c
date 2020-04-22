#include "ebms.h"

#include <omp.h>

/* There is only one memory group in this kernel */

#define FINE_TIME 0
#define FETCH_TIME 1
#define DEBUG_INFO 0

int main(int argc, char **argv)
{
    int final_flag, provided;
    int rank, num_ranks;
    int num_threads;
    int num_workers;
    int i;
    int num_bands;

    size_t band_size;
    size_t band_size_per_rank;
    size_t band_memory_per_rank;

    char *bands_mem;
    char *band_buffer;

#if FINE_TIME
    double *t_get_threads, *t_flush_threads;
    double *t_get_workers, *t_flush_workers;
    double min_t_get, max_t_get, mean_t_get;
    double min_t_flush, max_t_flush, mean_t_flush;
    int *threads_get_count, *workers_get_count;
    int tot_get_count;
#elif FETCH_TIME
    double *t_fetch_threads;
    double *t_fetch_workers;
    double min_t_fetch, max_t_fetch, mean_t_fetch;
    int *threads_fetch_count, *workers_fetch_count;
    int tot_fetch_count;
#else
    double t1, t2;
#endif

    MPI_Win *bands_win;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); 

    num_threads = omp_get_max_threads();
    num_workers = num_threads * num_ranks;
    
    setup(rank, num_ranks, num_ranks, num_threads, argc, argv,
            &num_bands, &band_size,
            &final_flag);
    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }
#if DEBUG_INFO
    if (rank == 0) printf("Number of threads %d\n", num_threads);
#endif

    /* The amount of bytes stored per band on each rank */
    band_size_per_rank = band_size / num_ranks;

    /* Allocate the memory for the bands on this rank.
     * This memory can be accessed directly by threads on this node
     * OR this memory can be accessed through RMA by threads on other nodes. */
    band_memory_per_rank = band_size_per_rank * num_bands;
    bands_mem = (char *) calloc(band_memory_per_rank, sizeof(char));

    bands_win = (MPI_Win *) malloc(num_threads * sizeof(MPI_Win));
    for (i = 0; i < num_threads; i++) {
        MPI_Win_create(bands_mem, band_memory_per_rank, sizeof(char),
                MPI_INFO_NULL, MPI_COMM_WORLD, &bands_win[i]);
    }

    /* Allocate the receive buffer for 1 band.*/
    band_buffer = (char*) malloc(band_size * sizeof(char));
   
#if FINE_TIME
    t_get_threads = calloc(num_threads, sizeof(double));
    t_flush_threads = calloc(num_threads, sizeof(double));
    threads_get_count = calloc(num_threads, sizeof(int));
#elif FETCH_TIME
    t_fetch_threads = calloc(num_threads, sizeof(double));
    threads_fetch_count = calloc(num_threads, sizeof(int));
#endif

    for (i = 0; i < num_threads; i++) {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, bands_win[i]);
    }

#pragma omp parallel
    {
        int tid;
        int band_i;
        char *my_band_buffer;
        MPI_Aint disp;
        int get_i, num_gets;
        int offset_in_shared_buffer;
        int id_within_a_target_rank;
        int start_target_rank, target_rank;
        size_t buffer_size_per_thread, size_of_each_get;
        char *get_into_addr;
#if FINE_TIME
        int get_counter;
        double t_get_start, t_get;
        double t_flush_start, t_flush;
#elif FETCH_TIME
        int fetch_counter;
        double t_fetch_start, t_fetch;
#endif

        tid = omp_get_thread_num();

        if (num_threads < num_ranks) {
            /* Each thread on the node will get from more than 1 rank */
            num_gets = num_ranks / num_threads;
            size_of_each_get = band_size_per_rank;
            id_within_a_target_rank = 0;
            start_target_rank = num_gets * tid;
        } else {
            /* Each thread will get only once. Concurrent gets
             * to the same target rank will have different offsets */
            num_gets = 1;
            size_of_each_get = band_size / num_threads;
            id_within_a_target_rank = tid % (num_threads / num_ranks);
            start_target_rank = tid / (num_threads / num_ranks);
        }
#if DEBUG_INFO
        if (rank == 0 && tid == 0) printf("Size of each get: %d\n", (int) size_of_each_get);
        if (rank == 0 && tid == 0) printf("Num gets: %d\n", num_gets);
        printf("Thread %d of Rank %d: ID within a target node %d\n", tid, rank, id_within_a_target_rank);
        printf("Thread %d of Rank %d: Start target rank %d\n", tid, rank, start_target_rank);
#endif
        
        buffer_size_per_thread = num_gets * size_of_each_get;
        offset_in_shared_buffer = tid * buffer_size_per_thread;

        /* First touch my share of the buffer */
        my_band_buffer = band_buffer + offset_in_shared_buffer;
        memset(my_band_buffer, 0, buffer_size_per_thread);

#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
#if DEBUG_INFO
            if (rank == 0) {
                printf("About to start!\n");
                fflush(stdout);
            }
#endif
        }
#pragma omp barrier
#if FINE_TIME
        get_counter = 0;
        t_get = t_flush = 0;
#elif FETCH_TIME
        fetch_counter = 0;
        t_fetch = 0;
#else
#pragma omp master
        {
            t1 = MPI_Wtime();
        }
#endif
        for (band_i = 0; band_i < num_bands; band_i++) {
            disp = (band_i * band_size_per_rank) + (id_within_a_target_rank * size_of_each_get);
#if DEBUG_INFO
            printf("Thread %d of Rank %d: Disp for band %d is %d\n", tid, rank, band_i, disp);
#endif
            /* Fetch my share of this band and store it in my part of the shared memory buffer */
            for (get_i = 0; get_i < num_gets; get_i++) {
                get_into_addr = my_band_buffer + (get_i * size_of_each_get);
                target_rank = start_target_rank + get_i;
                if (rank == target_rank) {
                    /* TODO: copy from this rank's memory */
                } else {
                    /* Get from remote node */
#if DEBUG_INFO
                    printf("Thread %d of Rank %d getting from rank %d\n", tid, rank, target_rank);
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
                        bands_win[tid]);
#if FINE_TIME
                    t_get += (MPI_Wtime() - t_get_start);
                    t_flush_start = MPI_Wtime();
#endif
                    MPI_Win_flush(target_rank, bands_win[tid]);
#if FINE_TIME
                    t_flush += (MPI_Wtime() - t_flush_start);
#endif
#if FETCH_TIME
                    t_fetch += (MPI_Wtime() - t_fetch_start);
#endif
                }
            /* Perform computation cooperatively with other threads using the fetched band data */
#pragma omp barrier
            }
        }

#if FINE_TIME
        threads_get_count[tid] = get_counter;
        if (get_counter > 0) {
            t_get_threads[tid] = t_get / get_counter;
            t_flush_threads[tid] = t_flush / get_counter;
        }
#elif FETCH_TIME
        threads_fetch_count[tid] = fetch_counter;
        if (fetch_counter > 0)
            t_fetch_threads[tid] = t_fetch / fetch_counter;
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD); 

#if FINE_TIME
    if (rank == 0) {
        t_get_workers = calloc(num_workers, sizeof(double));
        t_flush_workers = calloc(num_workers, sizeof(double));
        workers_get_count = calloc(num_workers, sizeof(int));
        if (!t_get_workers || !t_flush_workers || !workers_get_count) {
            fprintf(stderr, "Unable to allocate memory for t_get_workers or t_flush_workers or workers_get_count\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else { 
        t_get_workers = NULL;
        t_flush_workers = NULL;
        workers_get_count = NULL;
    }

    MPI_Gather(t_get_threads, num_threads, MPI_DOUBLE, t_get_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(t_flush_threads, num_threads, MPI_DOUBLE, t_flush_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* The reduce causes a bug: it changes the rank of the root to rank 3 with 4 ranks */
    //MPI_Reduce(threads_get_count, &tot_get_count, num_threads, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(threads_get_count, num_threads, MPI_INT, workers_get_count, num_threads, MPI_INT, 0, MPI_COMM_WORLD);

    mean_t_get = mean_t_flush = 0;
    max_t_get = max_t_flush = -1;
    min_t_get = min_t_flush = 9999;
    tot_get_count = 0;

    if (rank == 0) {
        int pi;
        double sum_t_get, sum_t_flush;
        int nworkers_who_got;

        nworkers_who_got = 0;
        sum_t_get = sum_t_flush = 0;

        for (pi = 0; pi < num_workers; pi++) {
            if (t_get_workers[pi] > 0) {
                if (!(t_flush_workers[pi] > 0))
                    printf("THIS SHOULD NEVER HAPPEN!\n");
                
                nworkers_who_got++;
                
                if (max_t_get < t_get_workers[pi])
                    max_t_get = t_get_workers[pi];
                if (min_t_get > t_get_workers[pi])
                    min_t_get = t_get_workers[pi];
                
                if (max_t_flush < t_flush_workers[pi])
                    max_t_flush = t_flush_workers[pi];
                if (min_t_flush > t_flush_workers[pi])
                    min_t_flush = t_flush_workers[pi];

                sum_t_get += t_get_workers[pi];
                sum_t_flush += t_flush_workers[pi];
            }
            tot_get_count += workers_get_count[pi];
        }
        mean_t_get = sum_t_get / nworkers_who_got;
        mean_t_flush = sum_t_flush / nworkers_who_got;
    }
#elif FETCH_TIME
    if (rank == 0) {
        t_fetch_workers = calloc(num_workers, sizeof(double));
        workers_fetch_count = calloc(num_workers, sizeof(int));
        if (!t_fetch_workers || !workers_fetch_count) {
            fprintf(stderr, "Unable to allocate memory for t_fetch_workers or workers_fetch_count\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else { 
        t_fetch_workers = NULL;
        workers_fetch_count = NULL;
    }

    MPI_Gather(t_fetch_threads, num_threads, MPI_DOUBLE, t_fetch_workers, num_threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* The reduce causes a bug: it changes the rank of the root to rank 3 with 4 ranks */
    //MPI_Reduce(threads_fetch_count, &tot_fetch_count, num_threads, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(threads_fetch_count, num_threads, MPI_INT, workers_fetch_count, num_threads, MPI_INT, 0, MPI_COMM_WORLD);

    mean_t_fetch = 0;
    max_t_fetch = -1;
    min_t_fetch = 9999;
    tot_fetch_count = 0;

    if (rank == 0) {
        int pi;
        double sum_t_fetch;
        int nworkers_who_fetched;

        nworkers_who_fetched = 0;
        sum_t_fetch = 0;

        for (pi = 0; pi < num_workers; pi++) {
            if (t_fetch_workers[pi] > 0) {
                nworkers_who_fetched++;
                
                if (max_t_fetch < t_fetch_workers[pi])
                    max_t_fetch = t_fetch_workers[pi];
                if (min_t_fetch > t_fetch_workers[pi])
                    min_t_fetch = t_fetch_workers[pi];
                
                sum_t_fetch += t_fetch_workers[pi];
            }
            tot_fetch_count += workers_fetch_count[pi];
        }
        mean_t_fetch = sum_t_fetch / nworkers_who_fetched;
    }
#else
    t2 = MPI_Wtime();
#endif
    
    for (i = 0; i < num_threads; i++) {
        MPI_Win_unlock_all(bands_win[i]);
    }

    if (rank == 0) {
#if FINE_TIME
        printf("num_bands,band_size,nworkers,min_get_time,max_get_time,mean_get_time,tot_get_count,min_flush_time,max_flush_time,mean_flush_time\n");
        printf("%d,%d,%d,%.9f,%.9f,%.9f,%d,%.9f,%.9f,%.9f\n", num_bands, (int) band_size, num_workers, min_t_get, max_t_get, mean_t_get, tot_get_count, min_t_flush, max_t_flush, mean_t_flush);
#elif FETCH_TIME
        printf("num_bands,band_size,nworkers,min_fetch_time,max_fetch_time,mean_fetch_time,tot_fetch_count\n");
        printf("%d,%d,%d,%.9f,%.9f,%.9f,%d\n", num_bands, (int) band_size, num_workers, min_t_fetch, max_t_fetch, mean_t_fetch, tot_fetch_count);
#else
        printf("num_bands,band_size,nworkers,time\n");
        printf("%d,%d,%d,%.9f\n", num_bands, (int) band_size, num_workers, t2-t1);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    for (i = 0; i < num_threads; i++) {
        MPI_Win_free(&bands_win[i]);
    }

    free(bands_win);
    free(bands_mem);
    free(band_buffer);
#if FINE_TIME
    free(t_get_threads);
    free(t_flush_threads);
    free(t_get_workers);
    free(t_flush_workers);
    free(threads_get_count);
    free(workers_get_count);
#endif
#if FETCH_TIME
    free(t_fetch_threads);
    free(t_fetch_workers);
    free(threads_fetch_count);
    free(workers_fetch_count);
#endif

    MPI_Finalize();
    return 0;
}

/******************************************************************************************************************
*       Title:          Micro-benchmark with broadcast based on the wrapper interfaces
*       Date:           01/04/2020
*       Authors:        Huan Zhou (huan.zhou@hlrs.de)
*       Institute:      High Performance Computing Center Stuttgart (HLRS)
******************************************************************************************************************/
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "wrapper.h"

#define MAX_SIZE (1<<15)
#define VALIDATE
//#define DEBUG

double standard_deviation(double data[], int n) {
    double mean = 0.0, sum_deviation = 0.0;
    int i;
    for(i = 0; i < n; i++)
        mean += data[i];
    mean = mean/n;

    for(i = 0; i < n; i++)
    	sum_deviation += (data[i] - mean) * (data[i] - mean);
    return sqrt(sum_deviation/(n-1));
}

int main (int argc, char *argv[])
{
    double      *s_buf=NULL, *r_buf=NULL;
    double      *result_addr;
    int         rank, nprocs, i, j, k = 0;
    //int         skip = 40;
    //int         loop = 300, inner_loop = 300, w_loop;
    int         skip = 40;
    int         loop = 600, inner_loop = 600, w_loop;
    double      t_start = 0.0, t_end = 0.0, t_total_dur = 0.0, *t_dur;
    double 	    t_avg[16], std_dev[16];
    struct comm_package comm_handle;
    int         sharedmem_rank;
    int         *bridge_transtable, *shmem_transtable;

    MPI_Win     win;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    Wrapper_MPI_ShmemBridgeComm_create(MPI_COMM_WORLD, &comm_handle);

    bridge_transtable = (int*)malloc (sizeof(int)*nprocs);
    shmem_transtable = (int*)malloc (sizeof(int)*nprocs);

    Wrapper_Get_transtable (MPI_COMM_WORLD, &comm_handle, &shmem_transtable, &bridge_transtable);

    t_dur = (double*)malloc(sizeof(double)*loop);
    for (int msg = 1; msg <= MAX_SIZE; msg*=2)
    {
        Wrapper_MPI_Sharedmemory_alloc(msg, sizeof(double), 0, &comm_handle, (void**)&s_buf, &win);
        MPI_Comm_rank (comm_handle.shmem_comm, &sharedmem_rank);

        if (rank == 0)
        {
            for (i = 0; i < msg; i++)
                s_buf[i] = i;
        }

        for (i = 0; i < skip; i++)
            Wrapper_Hy_bcast<double>(&r_buf, s_buf, msg, shmem_transtable,
               bridge_transtable, MPI_DOUBLE, 0, &comm_handle);

        t_total_dur = 0.0;
        for (i = 0; i < loop; i++)
        {
            t_start = MPI_Wtime();
            for (j = 0; j < inner_loop; j++) {
                Wrapper_Hy_bcast<double>(&r_buf, s_buf, msg, shmem_transtable,
                   bridge_transtable, MPI_DOUBLE, 0, &comm_handle);
            }
            t_end = MPI_Wtime();
            t_dur[i] = (t_end - t_start) * 1000000/inner_loop;
            t_total_dur += t_dur[i];
        }

        t_avg[k] = t_total_dur/loop;
        std_dev[k] = standard_deviation(t_dur, loop)/loop;
        MPI_Barrier (MPI_COMM_WORLD);

        double max_t;
        MPI_Reduce(&t_avg[k], &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            fprintf (stdout, "%d %f with deviation %f\n", msg, max_t, std_dev[k]);

    #ifdef VALIDATE
        int         size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        for (i = 0; i < msg; i++)
        {
            #ifdef DEBUG
            fprintf (stdout, "i %d msg: %d, %f\n", i, msg, result_addr[i]);
            #endif

            if (r_buf[i] != i)
                printf("error: %d , the value is %f\n", i, r_buf[i]);
        }
    #endif

        MPI_Barrier (MPI_COMM_WORLD);
        MPI_Win_free(&win);
        k++;
    }

    free (t_dur);
    free (bridge_transtable);
    free (shmem_transtable);
    Wrapper_Comm_free(&comm_handle);
    MPI_Finalize();
}

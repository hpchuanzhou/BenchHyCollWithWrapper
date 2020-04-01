/******************************************************************************************************************
*       Title:          Micro-benchmark with hybrid MPI+MPI context-based allgather based on the wrapper interfaces
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

    size_t      size;
    int         rank, nprocs, i, j, k = 0;
    int         skip = 20;
    int         loop = 100, inner_loop = 100, w_loop;
    double      t_start = 0.0, t_end = 0.0, t_total_dur = 0.0, *t_dur;
    double 	    t_avg[16], std_dev[16];

    int sharedmem_size, sharedmem_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct comm_package comm_handle;
    struct allgather_param param_handle;
    int* sharedmem_sizeset;
    MPI_Win win;
    Wrapper_MPI_ShmemBridgeComm_create(MPI_COMM_WORLD, &comm_handle);
    Wrapper_ShmemcommSizeset_gather(&comm_handle, &sharedmem_sizeset);
    
    t_dur = (double *) malloc(loop * sizeof(double));
    for (i = 0; i < loop; i++)
        t_dur[i] = 0.0;

    for (int msg = 1; msg <= MAX_SIZE; msg *= 2)
    {
        Wrapper_MPI_Sharedmemory_alloc(msg, sizeof(double), nprocs, &comm_handle, (void**)&r_buf, &win);
        Wrapper_Create_Allgather_param(msg, &comm_handle, sharedmem_sizeset, &param_handle);
        Wrapper_Get_localpointer(r_buf, rank, msg*sizeof(double), (void**)&s_buf);

        for (i = 0; i < msg; i++)
            s_buf[i] = i + rank*msg;

        for (i = 0; i < skip; i++)
            Wrapper_Hy_Allgather<double>(r_buf, s_buf, msg, MPI_DOUBLE, &param_handle, &comm_handle);

        t_total_dur = 0.0;
        for (i = 0; i < loop; i++)
        {
            t_start = MPI_Wtime();
            for (j = 0; j < inner_loop; j++)
                Wrapper_Hy_Allgather<double>(r_buf, s_buf, msg, MPI_DOUBLE, &param_handle, &comm_handle);
            t_end = MPI_Wtime();
            t_dur[i] = (t_end - t_start) * 1000000/inner_loop;
            t_total_dur += t_dur[i];
        }

        t_avg[k]   = t_total_dur/loop;
        std_dev[k] = standard_deviation(t_dur, loop)/loop;

        if (rank == 0){
            fprintf (stdout, "%d %7.4f with deviation %f\n", msg, t_avg[k], std_dev[k]);
        }

        #ifdef VALIDATE

        for (i = 0; i < msg*nprocs; i++)
        {
        #ifdef DEBUG
            fprintf (stdout, "%d msg: %d, %f\n", i, msg, r_buf[i]);
        #endif
            if (r_buf[i] != i)
                fprintf(stdout, "error: %d, the value is %f\n", i, r_buf[i]);
        }
        #endif
        MPI_Barrier (MPI_COMM_WORLD);
        MPI_Win_free(&win);
        Wrapper_Param_Free(&comm_handle, &param_handle);
        k++;
    }

    Wrapper_ShmemcommSizeset_free(&comm_handle, sharedmem_sizeset);
    Wrapper_Comm_free(&comm_handle);
    free (t_dur);
    MPI_Finalize();
}

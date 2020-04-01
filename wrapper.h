/******************************************************************************************************************
*       Title:          Wrapper functions for hybrid MPI+MPI context-based broadcast, allreduce and allgather approaches.
*       Comments:       Implemented in the hybrid MPI+MPI SHM context
*       Date:           01/04/2020
*       Authors:        Huan Zhou (huan.zhou@hlrs.de)
*       Institute:      High Performance Computing Center Stuttgart (HLRS)
******************************************************************************************************************/

#include <mpi.h>

/** @brief Definition of the splited communicators 
*/
struct comm_package
{
	MPI_Comm shmem_comm; /* shard-memory communicator */
	MPI_Comm bridge_comm; /* bridge communicator */
	int      shmemcomm_size; /* the size of the shared-memory communicator */
	int      bridgecomm_size; /* the size of the bridge communicator */
};

/** @brief Definition of the parameters for MPI_Allgatherv.
*/
struct allgather_param
{
	int *recvcounts; /* receive counts */
	int *displs; /* displacements */
};

/** @brief Global objects for allreduce approach.
*/
double count;
#define SWITCH_MSG_BYTESIZE 2<<11

/* -- The common wrapper functions that are invoked before the execution of the collective operations -- */

/** @brief Two-level spliting of the given parent communicator.
*   @param[in]  p_comm Parent communicator
*   @param[out] comm_handle an instance of struct comm_handle
*/
void Wrapper_MPI_ShmemBridgeComm_create(MPI_Comm p_comm, struct comm_package* comm_handle)
{
	int sharedmem_rank;
	int leader = 0;
	
	MPI_Comm shmem_comm = MPI_COMM_NULL;
	MPI_Comm bridge_comm = MPI_COMM_NULL;
	int shmemcomm_size = 0;
	int bridgecomm_size = 0;

	MPI_Comm_split_type(p_comm, MPI_COMM_TYPE_SHARED, 1, MPI_INFO_NULL, &shmem_comm);
	MPI_Comm_rank(shmem_comm, &sharedmem_rank);

	MPI_Comm_size(shmem_comm, &shmemcomm_size);
	/* Leaders constitute bridge communicator, other non-leader on-node processes are called children */
	MPI_Comm_split(p_comm, (sharedmem_rank == leader) ? 0 : MPI_UNDEFINED, 0, &bridge_comm);

	if (bridge_comm != MPI_COMM_NULL)
		MPI_Comm_size(bridge_comm, &bridgecomm_size);
	MPI_Bcast(&bridgecomm_size, 1, MPI_INT, 0, shmem_comm);

    comm_handle->shmem_comm = shmem_comm;
    comm_handle->bridge_comm = bridge_comm;
    comm_handle->shmemcomm_size = shmemcomm_size;
    comm_handle->bridgecomm_size = bridgecomm_size;

}

/** @brief Allocation of shared-memory.
*   @param[in] msize Number of elments
*   @param[in] bsize Size of an element in bytes
*   @param[in] flag  Differentiation between collective operations
*   @param[in] comm_handle Splitted communicators
*   @param[out] shmem_addir The beginning address of the allocated shared-memory
*   @param[out] winPtr An MPI window object
*/
void Wrapper_MPI_Sharedmemory_alloc(int msize, int bsize, int flag, struct comm_package* comm_handle, void** shmem_addr, MPI_Win* winPtr)
{
	MPI_Win win;
	MPI_Aint seg_size;
	int disp_unit;
	char* temp_loc;

	if (comm_handle->shmem_comm != MPI_COMM_NULL)
	{
		if (comm_handle->bridge_comm != MPI_COMM_NULL)
		{
			if (!flag)
				MPI_Win_allocate_shared(bsize*msize*(comm_handle->shmemcomm_size + 2)+bsize, bsize, MPI_INFO_NULL, comm_handle->shmem_comm, shmem_addr, winPtr);
			else /* when flag is not 0, then the alloc is for allgather operation */
				MPI_Win_allocate_shared(bsize*msize*flag, bsize, MPI_INFO_NULL, comm_handle->shmem_comm, shmem_addr, winPtr);
		}
		else
		{
			MPI_Win_allocate_shared(0, bsize, MPI_INFO_NULL, comm_handle->shmem_comm, &temp_loc, winPtr);
			MPI_Win_shared_query(*winPtr, 0, &seg_size, &disp_unit, shmem_addr);
		}
		count = 0.0;
	}
}

/** @brief Each process gets a local pointer to its portion of data.
*   @param[in] start_addr The beginning address of the allocated shared-memory
*   @param[in] rank*dsise  Indicates the position of the local data
*   
*   @note sharedmem_rank for bcast, allreduce; myrank for allgather
*/
static inline void Wrapper_Get_localpointer(void* start_addr, int rank, int dsize, void** local_addr)
{
	*local_addr = start_addr+rank*dsize;
}

/* -- Wrapper functions for allgather approach -- */

/** @brief Each process collects the size of each shared-memory communicator.
*   @param[in] comm_handle 
*   @param[out] sharedmem_sizeset Stores the size of all shared-memory communicators
*/
void Wrapper_ShmemcommSizeset_gather(const struct comm_package* comm_handle, int **sharedmem_sizeset)
{
	if (comm_handle->bridge_comm != MPI_COMM_NULL)
	{
		MPI_Alloc_mem(sizeof(int)*(comm_handle->bridgecomm_size), MPI_INFO_NULL, sharedmem_sizeset);
		MPI_Allgather(&(comm_handle->shmemcomm_size), 1, MPI_INT, *sharedmem_sizeset, 1, MPI_INT, comm_handle->bridge_comm);
	}
}

/** @brief Create the parameters for the MPI_Allgatherv.
*   @param[in] msize Number of elements
*   @param[in] comm_handle 
*   @param[in] sharedmem_size
*   @param[out] param_handle An instance of struct allgather_param
*/
void Wrapper_Create_Allgather_param(int msize, const struct comm_package* comm_handle, int* sharedmem_sizeset, struct allgather_param* param_handle)
{	
	if (comm_handle->bridge_comm != MPI_COMM_NULL)
	{

		MPI_Alloc_mem(sizeof(int)*(comm_handle->bridgecomm_size), MPI_INFO_NULL, &(param_handle->recvcounts));
		MPI_Alloc_mem(sizeof(int)*(comm_handle->bridgecomm_size), MPI_INFO_NULL, &(param_handle->displs));

		for (int i = 0; i < comm_handle->bridgecomm_size; i++)
		{
			(param_handle->recvcounts)[i] = msize*(sharedmem_sizeset[i]);
		}

		for (int i = 0; i < comm_handle->bridgecomm_size; i++)
		{
			(param_handle->displs)[i] = 0;
			for (int j = 0; j < i; j++)
				(param_handle->displs)[i] += msize*(sharedmem_sizeset)[j];
		}
	}
}

/** @brief allgather wrapper function corresponds to the standard allgather function.
*   All the parameters are input
*/
template <class myType>
void Wrapper_Hy_Allgather(myType* start_addr, myType* local_addr, int msize, MPI_Datatype data_type, struct allgather_param* param_handle,
	struct comm_package* comm_handle)
{
	if (comm_handle->bridge_comm != MPI_COMM_NULL)
	{ //leader
		if (comm_handle->bridgecomm_size > 1)
		{   
			/* on more than one node */
			if (comm_handle->shmemcomm_size > 1)
			{ 
				MPI_Barrier(comm_handle->shmem_comm);
			}
			/* allgather across nodes */
			MPI_Allgatherv(local_addr, msize*(comm_handle->shmemcomm_size), data_type,
				start_addr, param_handle->recvcounts, param_handle->displs, data_type, comm_handle->bridge_comm);
			if (comm_handle->shmemcomm_size > 1)
			{
				MPI_Barrier(comm_handle->shmem_comm);
			}
	    }
		else
		{
			if (comm_handle->shmemcomm_size > 1)
			{
				MPI_Barrier(comm_handle->shmem_comm);
			}
		}
	}
	else
	{// Children
		if (comm_handle->bridgecomm_size > 1)
		{
			if (comm_handle->shmemcomm_size > 1)
			{
				MPI_Barrier(comm_handle->shmem_comm);
				MPI_Barrier(comm_handle->shmem_comm);
			}
		}
		else
		{
			if (comm_handle->shmemcomm_size > 1)
			{
				MPI_Barrier(comm_handle->shmem_comm);
			}
		}
	}
}

/* -- Wrapper functions for broadcast approach -- */
/** @brief Create two absolute-to-relative rank translation tables
*   @param[in] p_comm Define the absolute rank in p_comm
*   @param[in] comm_handle Store the shared-memory and bridge communicators derived from p_comm
*   @param[out] shmem_transtable Map absolute rank in the p_comm to relative rank in the shared-memory communciator
*   @param[out] bridge_transtable Map absolute rank in the p_comm to relative rank in the bridge communicator
*/
void Wrapper_Get_transtable(MPI_Comm p_comm, const struct comm_package* comm_handle, int **shmem_transtable,
	int **bridge_transtable)
{
	MPI_Group group_all = MPI_GROUP_NULL;
	MPI_Group group_shmem = MPI_GROUP_NULL;
	MPI_Group group_bridge = MPI_GROUP_NULL;

	MPI_Comm_group(p_comm, &group_all);

	int comm_size;
	MPI_Comm_size(p_comm, &comm_size);

	int *all_ranks;
	all_ranks = (int*)malloc(sizeof(int)*comm_size);
	for (int i = 0; i < comm_size; i++)
		all_ranks[i] = i;

	MPI_Comm_group(comm_handle->shmem_comm, &group_shmem);
	MPI_Group_translate_ranks(group_all, comm_size, all_ranks, group_shmem, *shmem_transtable);

	if (comm_handle->bridge_comm != MPI_COMM_NULL)
	{
		MPI_Comm_group(comm_handle->bridge_comm, &group_bridge);
		MPI_Group_translate_ranks(group_all, comm_size, all_ranks, group_bridge, *bridge_transtable);

		for (int i = 1; i < comm_size; i++)
          if ((*bridge_transtable)[i] == MPI_UNDEFINED)
            (*bridge_transtable)[i] = (*bridge_transtable)[i-1];
	}
}

/** @brief broadcast wrapper function corresponds to the standard broadcast function.
*   All parameters except bcast_addr are input
*   @param[out] bcast_addr Returned address storing the broadcast data
*/
template <class myType>
void Wrapper_Hy_bcast(myType** bcast_addr, myType* start_addr, int msize,
	int* shmem_transtable, int* bridge_transtable, MPI_Datatype data_type, int root, struct comm_package* comm_handle)
{
	*bcast_addr = NULL;
	if ((comm_handle->shmem_comm != MPI_COMM_NULL) && (shmem_transtable[root] != MPI_UNDEFINED))
	{
		/* For the processes that are on the same node as root */
		(*bcast_addr) = start_addr + shmem_transtable[root] * msize;
	}
   
	if (comm_handle->bridgecomm_size > 1)
	{
		if ((*bcast_addr) == NULL) 
		{// The processes, that are on different node from root, use the hallo cell storing the broadcast data
			(*bcast_addr) = start_addr + comm_handle->shmemcomm_size * msize;
		}
		if (comm_handle->bridge_comm != MPI_COMM_NULL)
		{
			MPI_Bcast((*bcast_addr), msize, data_type, bridge_transtable[root], comm_handle->bridge_comm);
		}
	}

	if (shmem_transtable[root] == MPI_UNDEFINED)
		MPI_Barrier(comm_handle->shmem_comm);
}

/* -- Wrapper functions for allreduce approach -- */
/** allreduce wrapper function corresponds to the standard allreduce function.
*   All parameters except result_addr are input
*   @param[out] result_addr Returned address storing the reduced result
*/
// This is for the operation of summation (MPI_SUM)
template <class myType>
void Wrapper_Hy_Allreduce(myType* start_addr, myType** result_addr, int sharedmem_rank, int msize,
	MPI_Datatype data_type, MPI_Op op, struct comm_package* comm_handle, MPI_Win win)
{
    int threshold_size = msize*sizeof(myType);
    int bound_size = comm_handle->shmemcomm_size*msize;
	/* Compute the locally reduced result on each node */
    if (comm_handle->shmem_comm != MPI_COMM_NULL)
	{
		if (threshold_size <= SWITCH_MSG_BYTESIZE)
		{ // method 2: serial but not sync
			MPI_Barrier(comm_handle->shmem_comm);
			double* sum_addr;
			if (sharedmem_rank == 0)
			{
				for (int j = 0; j < msize; j++)
				{
					sum_addr = start_addr + bound_size + j;
					*sum_addr = (myType)0;
					for (int i = j; i < bound_size; i+=msize)
					{	
						*sum_addr += *(start_addr+i);
					}
				}
			}
		}
		else
		{ // method 1: parallel but induce implicit sync
			MPI_Reduce(start_addr + sharedmem_rank*msize, start_addr + bound_size, 
				msize, data_type, op, 0, comm_handle->shmem_comm);
		}
		count += 1.0;
	}

    if (comm_handle->bridgecomm_size <= 1) // on a single node
    	*result_addr = start_addr + bound_size;
    else // comm_handle->bridgecomm_size > 1, on more than one node
	{
		*result_addr = start_addr + bound_size + msize;
		if (comm_handle->bridge_comm != MPI_COMM_NULL)
		{
			/* Get the final reduced result across nodes */
			MPI_Allreduce(start_addr + bound_size, 
				*result_addr, msize, data_type, op, comm_handle->bridge_comm);
		}

	}

    /* Sync: the last element serves for this spinning method */
	myType* tmp_addr = start_addr + bound_size + 2*msize;
	if(sharedmem_rank == 0)
	{
		tmp_addr[0] += 1.0;
		MPI_Win_sync(win);
	}
	else
	{
		while (1)
		{
			MPI_Win_sync(win);
			if (tmp_addr[0] == count)
				break;
		}
	}
}

/* -- Deallocation -- */

/* -- Specific for allgather -- */
static inline void Wrapper_ShmemcommSizeset_free(struct comm_package* comm_handle, int* sharedmem_sizeset)
{
	if (comm_handle->bridge_comm != MPI_COMM_NULL)
		MPI_Free_mem(sharedmem_sizeset);
}

static inline void Wrapper_Param_Free(struct comm_package* comm_handle, struct allgather_param* param_handle)
{
	if (comm_handle->bridge_comm != MPI_COMM_NULL)
	{
		MPI_Free_mem(param_handle->recvcounts);
		MPI_Free_mem(param_handle->displs);
	}
}

/** -- Common deallocation -- **/
static inline void Wrapper_Comm_free(struct comm_package* comm_handle)
{
	MPI_Comm_free(&(comm_handle->shmem_comm));
	if (comm_handle->bridge_comm != MPI_COMM_NULL)
		MPI_Comm_free(&(comm_handle->bridge_comm));
}



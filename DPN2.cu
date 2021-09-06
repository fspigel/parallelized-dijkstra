#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define BLOCK_WIDTH 128
#define MAX_GRID_WIDTH 65535

bool debug = false;

//print a list of numbers
void Printv(int n, double * l){
	if(n > 10) n = 10;
	for(int i = 0; i < n-1; ++i) printf("%.0f, ", l[i]);
	printf("%.0f\n", l[n-1]);
}
void Printv(int n, unsigned long long int * l){
	if(n > 10) n = 10;
	for(int i = 0; i < n-1; ++i) printf("%llu, ", l[i]);
	printf("%llu\n", l[n-1]);
}
__device__ void Printv_d(int n, double * l){
	if(n > 10) n = 10;
	for(int i = 0; i < n-1; ++i) printf("%.0f, ", l[i]);
	printf("%.0f\n", l[n-1]);
}
__device__ void Printv_d(int n, unsigned long long int * l){
	if(n > 10) n = 10;
	for(int i = 0; i < n-1; ++i) printf("%llu, ", l[i]);
	printf("%llu\n", l[n-1]);
}

bool cudaErrorCheck(){
	cudaError_t cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess){
		printf("KERNEL ERROR!\n");
		printf(cudaGetErrorName(cudaError));
		printf("\n");
		printf(cudaGetErrorString(cudaError));
		printf("\n\n");
		return true;
	}
	return false;
}

//obtain a global linearized index, unique to all threads
__device__ unsigned long long int GetGlobalIdx_d(){
	unsigned long long int bID;
	bID = 	gridDim.x * gridDim.y * blockIdx.z +
		gridDim.x * blockIdx.y +
		blockIdx.x;
			
	unsigned long long int gID;
	gID = bID * blockDim.x + threadIdx.x;
	return gID;
}

//obtain grid dimensions necessary to launch n threads
uint3 GetGridDimensions(unsigned long long int n){
	unsigned long long int totalBlocks = ceil((double)n / BLOCK_WIDTH); //total number of blocks necessary to complete the task
	
	if(totalBlocks < MAX_GRID_WIDTH)
		return make_uint3(totalBlocks, 1, 1);
	if(totalBlocks < (unsigned long long int) MAX_GRID_WIDTH * MAX_GRID_WIDTH)
		return make_uint3(MAX_GRID_WIDTH, ceil(totalBlocks / MAX_GRID_WIDTH), 1);
	unsigned long long int M3 = (unsigned long long int)MAX_GRID_WIDTH * MAX_GRID_WIDTH * MAX_GRID_WIDTH;
	if(totalBlocks < M3)
		return make_uint3(MAX_GRID_WIDTH, MAX_GRID_WIDTH, ceil(totalBlocks / ((unsigned long long int)MAX_GRID_WIDTH * MAX_GRID_WIDTH)));
	
	printf("ERROR - PROBLEM SIZE EXCEEDS MAXIMUM OPERATIONAL PARAMETERS. UNABLE TO DETERMINE GRID DIMENSIONS. (n = %llu)\n", n);
	return make_uint3(0, 0, 0);
}

//initialize some arrays to starting values
__global__ void Initialize_g(	unsigned long long int n,
				unsigned long long int s,
				double * l,
				double * filteredL,
				unsigned long long int * filteredNodes,
				bool debug
				){
	unsigned long long int gID = GetGlobalIdx_d();
	
	if(gID == 0) printf("Initializing...\n");
	
	if(gID < n){
		if(gID != s){
			l[gID] = INFINITY;
			filteredL[gID] = INFINITY;
		}
		else{
			l[gID] = 0;
			filteredL[gID] = 0;
		}
		filteredNodes[gID] = gID;
	}
	
	if(debug){
		__syncthreads();
		if(gID == 0) Printv_d(n, filteredNodes);
	}	
}
	
	
//partially reduce an array - each block determines the minimum of a subsection
//of the array, then writes that minimum into a result array
__global__ void PartialReduce_g(	unsigned long long int n,
					double * work,			//the array which will be reduced
					unsigned long long int * nodes,	//keeps track of which element of work 
									//belongs to which node
					bool debug
					){
	//block index
	unsigned long long int bID = 	gridDim.x * gridDim.y * blockIdx.z +
					gridDim.x * blockIdx.y +
					blockIdx.x;
					
	//global thread index
	unsigned long long int gID = GetGlobalIdx_d();
	
	//local thread index within its block
	int lID = threadIdx.x;
	
	if(debug){
		if(gID == 0){
			printf("Commencing partial reduction.\nSize: %llu\nInitial data (first 10 elements):\n  array: ", n);
			Printv_d(n, work);
			printf("  nodes: ");
			Printv_d(n, nodes);
		}
	}
	
	//assign shared memory
	extern __shared__ double shared[];
	double * sharedDist = shared;
	unsigned long long int * sharedNodes = (unsigned long long int *)&sharedDist[BLOCK_WIDTH];
	
	if(gID < n){
		sharedDist[lID] = work[gID];
		sharedNodes[lID] = nodes[gID];
	}
	__syncthreads();
	
	//reduce internally
	for(int stride = BLOCK_WIDTH/2; stride >= 1; stride /= 2){
		if(lID + stride < n){
			if(sharedDist[lID] > sharedDist[lID + stride]){
				sharedDist[lID] = sharedDist[lID + stride];
				sharedNodes[lID] = sharedNodes[lID + stride];
			}
		}
		__syncthreads();
	}
	
	//write results
	if(lID == 0){
		work[bID] = sharedDist[0];
		nodes[bID] = sharedNodes[0];
	}
	
	if(debug){
		__syncthreads();	
		if(gID == 0){
			printf("Partial reduction complete.\nResults:\n  array: ");
			Printv_d(5, work);
			printf("  nodes: ");
			Printv_d(5, nodes);
		}
	}
}

//expand a node - first mark it as expanded, then update the distance vector with possible new distances
__global__ void Update_g(	double * E,				//weighted adjacency matrix
				unsigned long long int n,		//total number of nodes
				unsigned long long int fN,		//total number of remaining nodes
				double * l,				//an array denoting the distance between
									//the starting node, and already expanded nodes
				unsigned long long int * path,		//path vector with information about which path
									//to take from s to a given node
				unsigned long long int u,		//node scheduled for expansion
				double * filteredL,			//l with expanded nodes filtered out
				unsigned long long int * filteredNodes,	//node indices with already expanded nodes
									//filtered out		
				bool debug
				){
	//global index
	unsigned long long int gID = GetGlobalIdx_d();	
	
	//filtered distance value
	double fL;
	
	//filtered node index
	unsigned long long int fNode;
	
	if(debug){
		if(gID == 0){
			printf("Commencing Update on node %llu.\nInput data:\n  l: ", u);
			Printv_d(5, l);
			printf("  filteredL: ");
			Printv_d(fN, filteredL);
			printf("  filteredNodes: ");
			Printv_d(fN, filteredNodes);
			printf("\n");
		}
	}
		
	if(gID < fN){
		//assign fNode and fL
		fL = filteredL[gID];
		fNode = filteredNodes[gID];
		//assign l[u] its final value
		if(fNode == u) l[u] = fL;
	}
		
	__syncthreads();

	//update the distance vector with possible new distances
	if(gID < fN){
		double dist = l[u] + E[u*n + fNode];
		if(dist < fL){
			filteredL[gID] = dist;
			path[fNode] = u;
			fL = dist;
		}
	}
	
	if(gID < fN){
		//eliminate u from the list of unexpanded nodes by filtering it out
		//move all further elements back one spot
		if(fNode > u){
			filteredL[gID-1] = fL;
			filteredNodes[gID-1] = fNode;
		}
	}
	
	if(debug){
		__syncthreads();
		if(gID == 0){
			printf("Update complete. Resulting data:\n  l:");
			Printv_d(n, l);
			printf("  filteredL: ");
			Printv_d(fN, filteredL);
			printf("  filteredNodes: ");
			Printv_d(fN, filteredNodes);
		}
	}
}

void Dijkstra(	double * E,			//weighted adjacency matrix
		unsigned long long int n,	//total number of nodes
		unsigned long long int s,	//starting node
		double * l,			//distance vector such that l[u] = d(s, u)
		unsigned long long int * path	//path vector such that the shortest path 
						//from s to u goes through p[u]
		){
	printf("Commencing Dijkstra...\n_______________________________\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	
	//define and allocate memory for various arrays
	double *E_d, *l_d, *workL_d, *filteredL_d;
	unsigned long long int *path_d, *workNodes_d, *filteredNodes_d;
	
	cudaMalloc((void**) &E_d, n*n*sizeof(double));
	cudaMalloc((void**) &l_d, n*sizeof(double));
	cudaMalloc((void**) &workL_d, n*sizeof(double));
	cudaMalloc((void**) &filteredL_d, n*sizeof(double));
	
	cudaMalloc((void**) &path_d, n*sizeof(unsigned long long int));
	cudaMalloc((void**) &workNodes_d, n*sizeof(unsigned long long int));
	cudaMalloc((void**) &filteredNodes_d, n*sizeof(unsigned long long int));
	
	//transfer initial data to device memory
	cudaMemcpy(E_d, E, n*n*sizeof(double), cudaMemcpyHostToDevice);
	
	//initialize arrays
	printf("gridDim: (%u, %u, %u)\n", GetGridDimensions(n).x, GetGridDimensions(n).y, GetGridDimensions(n).z);
	Initialize_g<<<GetGridDimensions(n), BLOCK_WIDTH>>>(n, s, l_d, filteredL_d, filteredNodes_d, debug);
	cudaDeviceSynchronize();
	
	unsigned long long int sharedMemSize;
	double * tempL = (double*)malloc(sizeof(double));
	unsigned long long int * current = (unsigned long long int*)malloc(sizeof(unsigned long long int));
	*current = s;
		
	//commence the algorithm
	//complete an initial update by expanding the starting node s
	printf("Starting initial update - expanding node %llu\n", s); 
	Update_g<<<GetGridDimensions(n), BLOCK_WIDTH>>>(	E_d,
								n,
								n,
								l_d,
								path_d,
								s,
								filteredL_d,
								filteredNodes_d,
								debug
								);
	cudaDeviceSynchronize();
	if(cudaErrorCheck()) return;
	printf("Initial pdate completed successfully.\n");

	for(unsigned long long int i = 1; i < n; ++i){
		printf("Dijkstra iteration #%llu\n", i);
		
		//locate the next closest node via heavy reduction
		cudaMemcpy(workL_d, filteredL_d, (n-i)*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(workNodes_d, filteredNodes_d, (n-i)*sizeof(unsigned long long int), cudaMemcpyDeviceToDevice);
		for(unsigned long long j = n-i; j>0; j /= BLOCK_WIDTH){
			sharedMemSize = BLOCK_WIDTH * (sizeof(double) + sizeof(unsigned long long int));
			PartialReduce_g<<<GetGridDimensions(j), BLOCK_WIDTH, sharedMemSize>>>(	j, 
												workL_d, 
												workNodes_d,
												debug
												);
			cudaDeviceSynchronize();
			if(cudaErrorCheck()) return;
		}
		
		//reduction complete. closest node should now be workNodes_d[0]
		cudaMemcpy(tempL, workL_d, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(current, workNodes_d, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
		printf("Full reduction complete.\nClosest node: %llu; Distance: %.0f\n", *current, *tempL);
		printf("Expanding node %llu\n", *current);
		Update_g<<<GetGridDimensions(n-i), BLOCK_WIDTH>>>(	E_d,
									n,
									n-i,
									l_d,
									path_d,
									*current,
									filteredL_d,
									filteredNodes_d,
									debug
									);
		cudaDeviceSynchronize();
		if(cudaErrorCheck()) return;
		printf("Update completed successfully.\n\n");
	}
	
	cudaMemcpy(l, l_d, n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(path, path_d, n*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	
	printf("Algorithm complete. Results:\n  l: ");
	Printv(n, l);
	printf("  path: ");
	Printv(n, path);
	printf("\n");
}


int main(int argc, char *args[]){
	for(int i = 0; i < argc; ++i){
		if( strcmp(args[i], "-debug") == 0) debug = true;
	}

	unsigned long long int n = 5;
	
	double E[25] = {	0, 1, INFINITY, INFINITY, INFINITY,
				1, 0, 1, INFINITY, 10,
				INFINITY, 1, 0, INFINITY, 1,
				INFINITY, INFINITY, INFINITY, 0, 1,
				INFINITY, 10, 1, 1, 0
				};
	unsigned long long int s = 0;
	double * l = (double*)malloc(n*sizeof(double));
	unsigned long long int * path = (unsigned long long int*)malloc(n*sizeof(unsigned long long int));
	
	Dijkstra(E, n, s, l, path);
	
	return 0;
}

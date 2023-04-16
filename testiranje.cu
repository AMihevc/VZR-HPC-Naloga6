#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

// v pomoč pri debugganju kode
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


#define BINS 256
#define BLOCK_SIZE 32

void cumulative_histogram_cpu(unsigned int* h_hist_seq, unsigned int* cumulative, int bins) {
    
    // Calculate the cumulative histogram for each color
    for (int i = 0; i < bins; i++) {
        cumulative[i] = h_hist_seq[i] + ((i > 0) ? cumulative[i-1] : 0);
        cumulative[bins + i] = h_hist_seq[bins + i] + ((i > 0) ? cumulative[bins + i - 1] : 0);
        cumulative[2 * bins + i] = h_hist_seq[2 * bins + i] + ((i > 0) ? cumulative[2 * bins + i - 1] : 0);
    }

}

//gpu kernel for cumulative histogram calculation
__global__ void cumulative_histogram(unsigned int* d_histGPU, unsigned int* d_cumulative) {

    //get ids 
    int tid_global = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_local = threadIdx.x;
    int local_size = blockDim.x;
    int max = 0;

    //create a local sum array 
    // this can only be seen by threads within the same block
    __shared__ unsigned int localSum[BINS];

    //copy the histogram to the local sum array
    localSum[tid_local] = 0;
    if(tid_local < local_size){
        localSum[tid_local] = d_histGPU[tid_global];
    }

    //wait for all threads to finish copying (within a block)
    __syncthreads();

    // FAZA 1 - REDUCTION
    //reduce the local sum array to a single value
    for (int stride = 2; stride <= local_size; stride <<= 1) {

        if (tid_local % stride == stride - 1 ) {
            localSum[tid_local] += localSum[tid_local - stride / 2];
        }
        __syncthreads();
    }

    // resetiraj max vrednost ker se bo še enkrat zračunala v drugi fazi
    // to je trajal samo celo večnost da sm pogruntu zakaj je narobe:/ 
    if (tid_local == local_size - 1) {
        max = localSum[tid_local];
        localSum[tid_local] = 0;
    
    }

    // FAZA 2 - SCAN
    for (int stride = local_size; stride >= 2; stride >>= 1) {

        if (tid_local % stride == stride - 1 ) {
            int temp = localSum[tid_local]; // zapomni si vrednost na trenunti poziciji
            localSum[tid_local] += localSum[tid_local - stride / 2]; //seštej 
            localSum[tid_local - stride / 2] = temp; // prepiši vrednost na sešteti 
        }
        __syncthreads();
    }

    __syncthreads();

    // prepisocanje vrednosti iz lokalnega v globalni array
    //vse razn zadnje
    if(tid_local < local_size - 1){

        d_cumulative[tid_global] = localSum[tid_local + 1];
    }

    if (tid_local == local_size - 1) {
        d_cumulative[tid_global] = max;
    }
    __syncthreads();
    //TODO find samllest non zero in each BIN

    
}


int main() {
    
    // Declare and initialize the sequential histogram array
    unsigned int* h_hist_seq = (unsigned int*) calloc(3 * BINS, sizeof(unsigned int));
    
    // initilize the histogram with all ones for testing
    for (int i = 0; i < 3 * BINS; i++) {
        h_hist_seq[i] = 1;
    }
    
    // ################# CPU #################

    // version where the format is same as the histogram
    unsigned int* cumulative_one = (unsigned int*) calloc(3 * BINS, sizeof(unsigned int)); // Allocate memory for the output array

    // Call the function to calculate the cumulative histogram
    cumulative_histogram_cpu(h_hist_seq, cumulative_one, BINS);


    // ################# GPU #################
    
    // Declare and initialize the GPU histogram array
    // unsigned int *h_cumulative;     // cumulative histogram on host for copying to/from device
    unsigned int *d_cumulative;     // cumulative histogram on device
    unsigned int *d_histGPU;        // histogram on device
    
    // Allocate memory for the output array
    unsigned int* h_cumulative = (unsigned int*) calloc(3 * BINS, sizeof(unsigned int));

    // allocate and copy the histogram to the GPU
    checkCudaErrors(cudaMalloc(&d_cumulative, 3 * BINS* sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_cumulative, h_cumulative, 3 * BINS* sizeof(unsigned int), cudaMemcpyHostToDevice));

    // allocate and copy the histogram to the GPU
    checkCudaErrors(cudaMalloc(&d_histGPU, 3 * BINS* sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_histGPU, h_hist_seq, 3 * BINS* sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Set the thread execution grid (1 block)
    dim3 blockSize(256);

    // set grid size so we have 3 block one for each color
    dim3 gridSize(3);

    // call the kernel
    cumulative_histogram<<<gridSize, blockSize>>>(d_histGPU, d_cumulative);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy the histogram back to the CPU
    checkCudaErrors(cudaMemcpy(h_cumulative, d_cumulative, 3 * BINS* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    //printf("Copied mem from the GPU to the CPU\n");


    cudaFree(d_cumulative);
    cudaFree(d_histGPU);

    // ################# IZPISI #################

    // Print the histogram CPU
    // printf("HISTOGRAM CPU : \n");
    // for (int i = 0; i < BINS; i++) {
    //     printf("index: %d R: %d ", i, cumulative_one[i]);
    //     printf("G: %d ", cumulative_one[i + BINS]);
    //     printf("B: %d \n", cumulative_one[i + 2 * BINS]);
    // }

    // Print the histogram GPU
    // printf("HISTOGRAM GPU : \n");
    // for (int i = 0; i < BINS; i++) {
    //     printf("index: %d R: %d ", i,  h_cumulative[i]);
    //     printf("G: %d ", h_cumulative[i + BINS]);
    //     printf("B: %d \n", h_cumulative[i + 2 * BINS]);
    // }

    //chekc if CPU and GPU are the same
    int same = 1;
    for (int i = 0; i < 3 * BINS; i++) {
        if (cumulative_one[i] != h_cumulative[i]) {
            printf("ERROR: %d \n", i);
            same = 0;
        }
    }
    if (same) printf("SAME \n");


    // Print the cumulative histograms
    // printf("CUMULATIVE: \n");
    // for (int i = 0; i < BINS; i++) {
    //     printf("R: %d \n", r_cumulative[i]);
    //     printf("G: %d \n", g_cumulative[i]);
    //     printf("B: %d \n", b_cumulative[i]);
    // }

    //check if the cumulative histogram is the same as the one calculated in the function
    // int same = 1;
    // for (int i = 0; i < 3 * BINS; i++) {
    //     if (cumulative_one[i] != r_cumulative[i % BINS]) {
    //         printf("ERROR: %d \n", i);
    //         same = 0;
    //     }
    // }
    // if (same) printf("SAME \n");


    // ################# CLEAN UP #################

    // Free the memory allocated for the arrays
    free(h_hist_seq);
    free(cumulative_one);
    free(h_cumulative);


    return 0;
}

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

// Calc of of histogram equalization on the CPU
void CPU_Equalization(unsigned char* image, unsigned int* cdf, unsigned int* cdf_min,  int width, int height)
{
    // Calc equalization
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            int r = image[(i * width + j) * 4]; // RED
            int g = image[(i * width + j) * 4 + 1]; // GREEN
            int b = image[(i * width + j) * 4 + 2]; // BLUE
            double new_r = round(((double)((cdf)[r] - cdf_min[0]) / (double)(height*width - cdf_min[0])) * (BINS - 1));
            double new_g = round(((double)((cdf+BINS)[g] - cdf_min[1]) / (double)(height*width - cdf_min[1])) * (BINS - 1));
            double new_b = round(((double)((cdf+BINS*2)[b] - cdf_min[2]) / (double)(height*width - cdf_min[2])) * (BINS - 1));
            image[(i * width + j) * 4] = (unsigned char)new_r;
            image[(i * width + j) * 4 + 1] = (unsigned char)new_g;
            image[(i * width + j) * 4 + 2] = (unsigned char)new_b;
        }
}

//gpu kernel for cumulative histogram calculation
__global__ void cumulative_histogram(unsigned int* d_histGPU, unsigned int* d_cumulative, unsigned int* d_cdf_mins, int chanels) {

    //get ids 
    int tid_global = blockDim.x * blockIdx.x + threadIdx.x; // 0 - 3*256
    int tid_local = threadIdx.x; // 0-256
    int tid_block = blockIdx.x; // 0-3
    int local_size = blockDim.x; // 256
    int max = 0; //for max element in each bin

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

    if (tid_local == 0) {
        for (int i = 0; i < local_size; i++) {
            if (localSum[i] != 0) {
                d_cdf_mins[tid_block] = localSum[i];
                break;
            }
        }

    }


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
    unsigned int *h_cumulative;     // cumulative histogram on host for copying to/from device
    unsigned int *d_cumulative;     // cumulative histogram on device
    unsigned int *d_histGPU;        // histogram on device
    unsigned int *h_cdf_mins;         // minimum values of each bin on device
    unsigned int *d_cdf_mins;         // minimum values of each bin on device

    //allocate memory for cdf mins
    h_cdf_mins = (unsigned int*) calloc(3, sizeof(unsigned int));

    //allocate and copy the cdf mins to the GPU
    checkCudaErrors(cudaMalloc(&d_cdf_mins, 3 * sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_cdf_mins, h_cdf_mins, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    
    // Allocate memory for the output array
    h_cumulative = (unsigned int*) calloc(3 * BINS, sizeof(unsigned int));

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
    cumulative_histogram<<<gridSize, blockSize>>>(d_histGPU, d_cumulative, d_cdf_mins, 3);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // copy the cumulative histogram back to the CPU
    checkCudaErrors(cudaMemcpy(h_cumulative, d_cumulative, 3 * BINS* sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //copy the cdf mins back to the CPU
    checkCudaErrors(cudaMemcpy(h_cdf_mins, d_cdf_mins, 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

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

    //print cdf_mins
    printf("CDF MINS: \n");
    printf("R: %d G: %d B: %d \n", h_cdf_mins[0], h_cdf_mins[1], h_cdf_mins[2]);


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

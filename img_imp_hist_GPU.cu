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


// calculate histogram on CPU 
void histogramCPU(unsigned char *imageIn,
                  unsigned int *hist,
                  int width, int height, int cpp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            hist[imageIn[(i * width + j) * cpp]]++;                // RED
            hist[imageIn[(i * width + j) * cpp + 1] + BINS]++;     // GREEN
            hist[imageIn[(i * width + j) * cpp + 2] + 2 * BINS]++; // BLUE
        }
} // end of histogramCPU

// print the histogram by component (RED, GREEN, BLUE)
void printHistogram(unsigned int *hist)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < BINS; i++)
    {
        if (hist[i] > 0)
            printf("%dR\t%d\n", i, hist[i]);
        if (hist[i + BINS] > 0)
            printf("%dG\t%d\n", i, hist[i + BINS]);
        if (hist[i + 2 * BINS] > 0)
            printf("%dB\t%d\n", i, hist[i + 2 * BINS]);
    }
}


// GPU kernel to compute the histogram on the GPU
__global__ void histogramGPU(unsigned char* imageIn,
                             unsigned int* hist,
                             int width, int height, int cpp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms

    // calculate global and local id and use them to calculate the pixel index
    int tid_global_j = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_global_i = blockDim.y * blockIdx.y + threadIdx.y;

    // check that threads dont check pixels "outside" the image
    if ( tid_global_i < height && tid_global_j < width) {

        atomicAdd(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp]], 1);               // RED
        atomicAdd(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp + 1]+ BINS], 1);     // GREEN
        atomicAdd(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp + 2]+ 2*BINS], 1);   // BLUE
    }

}


// Sequential cumulative sum of the histogram
void cumulative_histogram_cpu(unsigned int* h_hist_seq, unsigned int* cumulative, int bins) {
    
    // Calculate the cumulative histogram for each color
    for (int i = 0; i < bins; i++) {
        cumulative[i] = h_hist_seq[i] + ((i > 0) ? cumulative[i-1] : 0);
        cumulative[bins + i] = h_hist_seq[bins + i] + ((i > 0) ? cumulative[bins + i - 1] : 0);
        cumulative[2 * bins + i] = h_hist_seq[2 * bins + i] + ((i > 0) ? cumulative[2 * bins + i - 1] : 0);
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
    
    // find samllest non zero in each BIN
    if (tid_local == 0) {
        for (int i = 0; i < local_size; i++) {
            if (localSum[i] != 0) {
                d_cdf_mins[tid_block] = localSum[i];
                break;
            }
        }
    }
}


//popravljanje slike 

// Calc of of histogram equalization on the CPU
void cpu_equalize(unsigned char* new_image_CPU, unsigned int* h_cumulative_seq, unsigned int* h_cdf_mins_seq,  int width, int height)
{
    int img_size = width * height;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++)
        {
            int color_offest = (i * width + j) * 4; // Red, +1 Green, +2 Blue
            //calc and save new values for each color
            new_image_CPU[color_offest] = (unsigned char)round(((double)(h_cumulative_seq[new_image_CPU[color_offest]] - h_cdf_mins_seq[0]) / (double)(img_size - h_cdf_mins_seq[0])) * (BINS - 1)); // RED
            new_image_CPU[color_offest + 1] = (unsigned char)round(((double)(h_cumulative_seq[new_image_CPU[color_offest + 1]+ BINS] - h_cdf_mins_seq[1]) / (double)(img_size - h_cdf_mins_seq[1])) * (BINS - 1)); // GREEN
            new_image_CPU[color_offest + 2] = (unsigned char)round(((double)(h_cumulative_seq[new_image_CPU[color_offest +2] + 2*BINS] - h_cdf_mins_seq[2]) / (double)(img_size - h_cdf_mins_seq[2])) * (BINS - 1)); // BLUE
            new_image_CPU[color_offest + 3] = 255; // Alpha
        }
    }
}

//gpu kernel for equalization
// GPU kernel to compute the histogram on the GPU
__global__ void gpu_equalize(unsigned char* new_image_CPU, unsigned int* d_hist_cumulative, unsigned int* d_cdf_mins ,int width, int height)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms

    // calculate global and local id and use them to calculate the pixel index
    int tid_global_j = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_global_i = blockDim.y * blockIdx.y + threadIdx.y;

    // check that threads dont check pixels "outside" the image
    if ( tid_global_i < height && tid_global_j < width) {

        int color_offest = (tid_global_i * width + tid_global_j) * 4; // Red, +1 Green, +2 Blue

        //calc and save new values for each color
        new_image_CPU[color_offest] = (unsigned char)round(((double)(d_hist_cumulative[new_image_CPU[color_offest]] - d_cdf_mins[0]) / (double)(width * height - d_cdf_mins[0])) * (BINS - 1)); // RED
        new_image_CPU[color_offest + 1] = (unsigned char)round(((double)(d_hist_cumulative[new_image_CPU[color_offest + 1]+ BINS] - d_cdf_mins[1]) / (double)(width * height - d_cdf_mins[1])) * (BINS - 1)); // GREEN
        new_image_CPU[color_offest + 2] = (unsigned char)round(((double)(d_hist_cumulative[new_image_CPU[color_offest +2] + 2*BINS] - d_cdf_mins[2]) / (double)(width * height - d_cdf_mins[2])) * (BINS - 1)); // BLUE
        new_image_CPU[color_offest + 3] = 255; // Alpha
    }

}


int main(int argc, char **argv)
{
    char *image_file = argv[1];

    // if image not provided exit
    if (argc > 1)
    {
        image_file = argv[1];
    }
    else
    {
        fprintf(stderr, "Not enough arguments\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(1);
    }

    // Initalize variables

    unsigned int *h_hist_seq;       // histogram on host for sequential computation
    unsigned int *h_hist;           // histogram on host for copying to/from device
    unsigned int *d_histGPU;        // histogram on device
    unsigned char *d_imageGPU;      // image on device
    int width, height, cpp;         // image properties

    unsigned int *h_cumulative_seq; // cumulative histogram on host for sequential computation
    unsigned int *h_cumulative;     // cumulative histogram on host for copying to/from device
    unsigned int *d_cumulative;     // cumulative histogram on device

    unsigned int *h_cdf_mins_seq;   // cdf mins on host for sequential computation TODO 
    unsigned int *h_cdf_mins;       // cdf mins on host for copying to/from device
    unsigned int *d_cdf_mins;       // cdf mins on device

    unsigned char* new_image_GPU;    // new image on device
    unsigned char* new_image_CPU;    // new image on host

    //time variables
    float milliseconds_CPU;
    float milliseconds_GPU;

    // load the image
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, 0); 
    
    // allocate memory for the histogram on the CPU
    h_hist_seq = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));
    
    //########## CPU ##########
    
    if (image_in)
    {
        // Compute and print the on CPU 

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // compute the histogram on the CPU
        histogramCPU(image_in, h_hist_seq, width, height, cpp);
        
        // compute the cumulative histogram on the CPU

        // Allocate memory for the output array
        h_cumulative_seq = (unsigned int*) calloc(3 * BINS, sizeof(unsigned int));

        // Call the function to calculate the cumulative histogram
        cumulative_histogram_cpu(h_hist_seq, h_cumulative_seq, BINS);

        // Find the smallest non-zero value in each bin

        // Allocate memory for the output array
        h_cdf_mins_seq = (unsigned int*) calloc(3, sizeof(unsigned int));
        int foundR= 0;
        int foundG = 0;
        int foundB = 0;
        for (int i = 0; i < BINS; i++) {
            if (h_cumulative_seq[i] != 0 && foundR == 0) {
                h_cdf_mins_seq[0] = h_cumulative_seq[i];
                foundR = 1;
            }
            if (h_cumulative_seq[i + BINS] != 0 && foundG == 0) {
                h_cdf_mins_seq[1] = h_cumulative_seq[i + BINS];
                foundG = 1;
            }
            if (h_cumulative_seq[i + 2 * BINS] != 0 && foundB == 0) {
                h_cdf_mins_seq[2] = h_cumulative_seq[i + 2 * BINS];
                foundB = 1;
            }
            if (foundR == 1 && foundG == 1 && foundB == 1) {
                break;
            }
        }

        // fix the picture
        int img_size = width * height;
        int width1, height1, cpp1;         // image properties 1

        new_image_CPU = (unsigned char *)malloc(img_size * 4 * sizeof(unsigned char));
        memcpy(new_image_CPU, image_in, img_size * 4 * sizeof(unsigned char));   

        // Apply the histogram equalization on the image
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++)
            {
                int color_offest = (i * width + j) * 4; // Red, +1 Green, +2 Blue
                //calc and save new values for each color
                new_image_CPU[color_offest] = (unsigned char)round(((double)(h_cumulative_seq[new_image_CPU[color_offest]] - h_cdf_mins_seq[0]) / (double)(img_size - h_cdf_mins_seq[0])) * (BINS - 1)); // RED
                new_image_CPU[color_offest + 1] = (unsigned char)round(((double)(h_cumulative_seq[new_image_CPU[color_offest + 1]+ BINS] - h_cdf_mins_seq[1]) / (double)(img_size - h_cdf_mins_seq[1])) * (BINS - 1)); // GREEN
                new_image_CPU[color_offest + 2] = (unsigned char)round(((double)(h_cumulative_seq[new_image_CPU[color_offest +2] + 2*BINS] - h_cdf_mins_seq[2]) / (double)(img_size - h_cdf_mins_seq[2])) * (BINS - 1)); // BLUE
                new_image_CPU[color_offest + 3] = 255; // Alpha
            }
        }

        // stop time measurement
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        milliseconds_CPU = 0;
        cudaEventElapsedTime(&milliseconds_CPU, start, stop);
        
        //printHistogram(h_hist_seq);
        printf("CPU time: %0.3f milliseconds \n", milliseconds_CPU);

        // save the image
        stbi_write_png("eqlized_cpu.png", width, height, cpp, new_image_CPU, width * cpp);
        stbi_write_jpg("eqlized_cpu2.jpg", width, height, cpp, new_image_CPU, 65);
        
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }
    
    
    //########## GPU ##########
    
    if (image_in)
    {
        // compute the histogram on the GPU

        // initialize the timig variables
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // allocate memory for the histogram, calcualted on the GPU, on the host 
        h_hist = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));

        // allocate and copy the histogram to the GPU
        checkCudaErrors(cudaMalloc(&d_histGPU, 3 * BINS* sizeof(unsigned int)));
        checkCudaErrors(cudaMemcpy(d_histGPU, h_hist, 3 * BINS* sizeof(unsigned int), cudaMemcpyHostToDevice));
        //printf("Allocated mem for the histogram on the GPU\n");

        //time mesurement start
        cudaEventRecord(start);
        
        // allocate and copy the image to the GPU
        int size = width * height * cpp * sizeof(unsigned char);
        checkCudaErrors(cudaMalloc(&d_imageGPU, size));
        checkCudaErrors(cudaMemcpy(d_imageGPU, image_in, size, cudaMemcpyHostToDevice));
        //printf("Allocated mem for the image on the GPU\n");

        // Set the thread execution grid (1 block)
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

        // calculate the grid size so that there is enough blocks to cover the whole image (1 pixel per thread)
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // call the kernel
        //printf("Launching the kernel\n");
        histogramGPU<<<gridSize, blockSize>>>(d_imageGPU, d_histGPU, width, height, cpp);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // copy the histogram back to the host
        checkCudaErrors(cudaMemcpy(h_hist, d_histGPU, 3 * BINS* sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //printf("Copied mem from the GPU to the CPU\n");
 
        // ############## CUMULATIVE GPU ##############
        
        //allocate memory for cdf mins
        h_cdf_mins = (unsigned int*) calloc(3, sizeof(unsigned int));

        //allocate and copy the cdf mins to the GPU
        checkCudaErrors(cudaMalloc(&d_cdf_mins, 3 * sizeof(unsigned int)));
        checkCudaErrors(cudaMemcpy(d_cdf_mins, h_cdf_mins, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        // Allocate memory for the output array
        h_cumulative = (unsigned int*) calloc(3 * BINS, sizeof(unsigned int));

        // allocate and copy the cumulative histogram to the GPU
        checkCudaErrors(cudaMalloc(&d_cumulative, 3 * BINS* sizeof(unsigned int)));
        checkCudaErrors(cudaMemcpy(d_cumulative, h_cumulative, 3 * BINS* sizeof(unsigned int), cudaMemcpyHostToDevice));

        // allocate and copy the histogram to the GPU (the hist should already be on the GPU from the previous kernel)
        // checkCudaErrors(cudaMalloc(&d_histGPU, 3 * BINS* sizeof(unsigned int)));
        // checkCudaErrors(cudaMemcpy(d_histGPU, h_hist_seq, 3 * BINS* sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Set the thread execution grid (1 block)
        dim3 blockSize2(BINS);

        // set grid size so we have 3 block one for each color
        dim3 gridSize2(3);

        // call the kernel
        cumulative_histogram<<<gridSize2, blockSize2>>>(d_histGPU, d_cumulative, d_cdf_mins, 3);

        err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // copy the cumulative histogram back to the CPU
        checkCudaErrors(cudaMemcpy(h_cumulative, d_cumulative, 3 * BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //copy the cdf mins back to the CPU
        checkCudaErrors(cudaMemcpy(h_cdf_mins, d_cdf_mins, 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // ############## EQUALIZATION GPU ##############
        gpu_equalize<<<gridSize, blockSize>>>(d_imageGPU, d_cumulative, d_cdf_mins, width, height);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //alocate memory for new image on the CPU and copy the equalized image back to the CPU
        new_image_GPU = (unsigned char*) malloc(size);

        // copy the equalized image back to the CPU
        checkCudaErrors(cudaMemcpy(new_image_GPU, d_imageGPU, size, cudaMemcpyDeviceToHost));

        //time mesurement stop
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        // Save the equalized image
        stbi_write_png("equalized_GPU.png", width, height, cpp, new_image_GPU, width * cpp);

        //free the GPU memory
        cudaFree(d_histGPU);
        cudaFree(d_imageGPU);
        cudaFree(d_cumulative);
        cudaFree(d_cdf_mins);

        // Display time mesurments
        float milliseconds_GPU = 0;
        cudaEventElapsedTime(&milliseconds_GPU, start, stop);
        printf("GPU time: %0.3f milliseconds \n", milliseconds_GPU);

        //print speed up
        printf("Speed up: %0.3f \n", milliseconds_CPU/milliseconds_GPU);

        // Display the histogram 
        //printHistogram(h_hist); 

        // Check if the histograms are the same
        int same = 0;  
        for (int i = 0; i < 3 * BINS; i++)
        {
            if (h_hist[i] != h_hist_seq[i])
            {
                printf("Histograms are not the same!\n");
                same = 1;
                break;
            }
        }
        if (!same)
        {
            printf("Histograms are the same!\n");
        }

        // check if the cumulative histograms are the same
        same = 1;
        for (int i = 0; i < 3 * BINS; i++)
        {
            if (h_cumulative[i] != h_cumulative_seq[i])
            {
                printf("Cumulative histograms are not the same!\n");
                same = 0;
                break;
            }
        }
        if(same)
        {
            printf("Cumulative histograms are the same!\n");
        }

        //print the cdf mins on CPU
        // printf("CDF MINS CPU : \n");
        // printf("R: %d G: %d B: %d \n", h_cdf_mins_seq[0], h_cdf_mins_seq[1], h_cdf_mins_seq[2]);
        // //print the cdf mins on GPU 
        // printf("CDF MINS GPU : \n");
        // printf("R: %d G: %d B: %d \n", h_cdf_mins[0], h_cdf_mins[1], h_cdf_mins[2]);

    } //end of hist on GPU
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }

    //########## CLEAN UP ##########

    // Free the image
    stbi_image_free(image_in);
    stbi_image_free(new_image_CPU);
    
    // Free the histograms
    free(h_hist);
    free(h_hist_seq);

    // Free the cumulative histogram
    free(h_cumulative_seq);
    free(h_cumulative);
    free(h_cdf_mins);
    free(h_cdf_mins_seq);

    
    return 0;
}
// CPU time: 1690.274 milliseconds 
// GPU time: 87.107 milliseconds 
// Speed up: 19.405 
// Histograms are the same!
// Cumulative histograms are the same!
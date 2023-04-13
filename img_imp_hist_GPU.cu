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
#define BLOCK_SIZE 32 //TODO this probably needs to be changed to 256 or 512



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


// a GPU kernel to compute the histogram on the GPU
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

        atomicAdd(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp]], 1);              // RED
        atomicAdd(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp + 1]+ BINS], 1);     // GREEN
        atomicAdd(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp + 2]+ 2*BINS], 1);   // BLUE
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
    unsigned int *h_hist;
    unsigned int *h_hist_seq;
    unsigned int *d_histGPU;
    unsigned char *d_imageGPU;
    int width, height, cpp;

    // load the image
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, 0); 
    
    // allocate memory for the histogram on the CPU
    h_hist_seq = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));
    
    //########## CPU ##########
    
    if (image_in)
    {
        // Compute and print the histogram

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        histogramCPU(image_in, h_hist_seq, width, height, cpp);
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        //printHistogram(h_hist_seq);
        printf("CPU time: %0.3f milliseconds \n", milliseconds);
        
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }
    
    
    
    //########## GPU ##########

    // Compute and print the histogram
    if (image_in) // if image loaded
    {
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

        // copy the histogram back to the CPU
        checkCudaErrors(cudaMemcpy(h_hist, d_histGPU, 3 * BINS* sizeof(unsigned int), cudaMemcpyDeviceToHost));
        //printf("Copied mem from the GPU to the CPU\n");

        //time mesurement stop
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        //free the GPU memory
        cudaFree(d_histGPU);
        cudaFree(d_imageGPU);

        // Display time mesurments
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU time: %0.3f milliseconds \n", milliseconds);

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
        if (same == 0)
        {
            printf("Histograms are the same!\n");
        }

    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }

    //########## CLEAN UP ##########

    // Free the image
    stbi_image_free(image_in);
    
    // Free the histograms
    free(h_hist);
    free(h_hist_seq);

    return 0;
}
/*
CPU time: 70.341 milliseconds 
GPU time: 40.059 milliseconds 
Histograms are the same!
*/
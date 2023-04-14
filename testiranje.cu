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




    // ################# IZPISI #################

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

    return 0;
}

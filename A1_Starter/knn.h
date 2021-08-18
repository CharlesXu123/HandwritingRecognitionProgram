#pragma once

/**
 * You will not be submitting this file, so do not change anything here
 * as it will not be reflected when the automarker is run. If you need any
 * extra structs / definitions, you can add them at the top of your `knn.c`
 * file, so they do not interfere with anything else.
 */

#include <math.h>    // Need this for sqrt()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This struct stores the data for an image */
typedef struct {
    int sx;               // x resolution
    int sy;               // y resolution
    unsigned char *data;  // List of `sx * sy` pixel color values [0-255]
} Image;

/* This struct stores the images / labels in the dataset */
typedef struct {
    int num_items;          // Number of images in the dataset
    Image *images;          // List of `num_items` Image structs
    unsigned char *labels;  // List of `num_items` labels [0-9]
} Dataset;

Dataset *load_dataset(const char *filename);
double distance(Image *a, Image *b);
int knn_predict(Dataset *data, Image *img, int K);
void free_dataset(Dataset *data);
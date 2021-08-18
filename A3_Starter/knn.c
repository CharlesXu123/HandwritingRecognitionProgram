#include "knn.h"

/****************************************************************************/
/* For all the remaining functions you may assume all the images are of the */
/*     same size, you do not need to perform checks to ensure this.         */
/****************************************************************************/

/**************************** A1 code ****************************************/

/* Same as A1, you can reuse your code if you want! */
double distance(Image *a, Image *b) {
  int size = a->sx*a->sy;
  double sum_dist = 0.0;
  for (int i = 0; i < size; i++){
    int diff = ((a->data[i]) - (b->data[i]));
    sum_dist = sum_dist + diff*diff;
  }
  return sqrt(sum_dist);
  return 0; 
}

/*this struct store labels and distance of an Image */
typedef struct {
  unsigned char label;
  double distance;
} KNode;

void swap_larg(KNode* k_array, int K)
{
  int index_larg = 0;
  double larg = -1.0;
  for (int i = 0; i<K; i++){
    if ((k_array[i].distance)>larg){
      index_larg = i;
      larg = k_array[i].distance;
    }
  }
  unsigned char lab = k_array[index_larg].label;
  k_array[index_larg].distance = k_array[K-1].distance;
  k_array[index_larg].label = k_array[K-1].label;
  k_array[K-1].distance = larg;
  k_array[K-1].label = lab;
}

/* Same as A1, you can reuse your code if you want! */
int knn_predict(Dataset *data, Image *input, int K) {
  //construct a KNode array of size k
  KNode k_array[K];
  for (int i = 0; i<K; i++){
    k_array[i].label = 0;
  }
  int count = 0;
  //fill the k_array with KNode with smallest distance
  for (int i = 0; i < data->num_items; i++){
    double dist = distance(input, &(data->images[i]));
    //printf("listofdist%d: %f\n", i, dist);
    unsigned char lab = data->labels[i];
    if (count < K){
      k_array[count].distance = dist;
      k_array[count].label = lab;
      count++;
    }
    else{
      swap_larg(k_array, K);
      if (dist < (k_array[K-1].distance)){
        k_array[K-1].distance = dist;
        k_array[K-1].label = lab;
      }
    }
    }
  //find the most frequent label
  int labels[10];
  int times = 0;
  int index_of_predict = 0;
  for (int i = 0; i<10; i++){
    labels[i] = 0;
  }
  for (int i = 0; i<K; i++){
    //printf("dist: %f ", k_array[i]->distance);
    labels[k_array[i].label]++;
  }
  //printf("\nlist: ");
  for (int i = 0; i<10; i++){
    //printf("%d ", labels[i]);
    if (labels[i] >= times){
      times = labels[i];
      index_of_predict = i;
    }
  }
  return index_of_predict;
}

/**************************** A2 code ****************************************/

/* Same as A2, you can reuse your code if you want! */
/**
 * This function is the helper function of load_dataset
 */
void read_file(const char *filename, Dataset* dataset){
  FILE* file = fopen(filename, "rb");
  if (file == NULL){
    fprintf(stderr, "Error: could not open %s\n", filename);
    return;
  }
  int num_items;
  short error = fread(&num_items, 4, 1, file);
  if (error == 0){
    fprintf(stderr, "Error: could not read %s\n", filename);
    return;
  }
  dataset->num_items = num_items;
  unsigned char label;
  dataset->labels = (unsigned char*)malloc(num_items*sizeof(unsigned char));
  dataset->images = (Image*)malloc(num_items*sizeof(Image));
  int i = 0;
  while (fread(&label, 1, 1, file) == 1){
    dataset->labels[i] = label;
    dataset->images[i].sx = 28;
    dataset->images[i].sy = 28;
    dataset->images[i].data = (unsigned char*)malloc(784*sizeof(unsigned char));
    if (fread(dataset->images[i].data, 1, 784, file) == 0){
      perror("fread");
      exit(1);
    }
    i++;
  }
  fclose(file);
}

/**
 * This function takes in the name of the binary file containing the data and
 * loads it into memory. The binary file format consists of the following:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - 784 bytes : Image 1 data (28x28)
 *          ...
 *     -   1 byte  : Image N label
 *     - 784 bytes : Image N data (28x28)
 *
 * You can simply set the `sx` and `sy` values for all the images to 28, we
 * will not test this with different image sizes.
 */
Dataset *load_dataset(const char *filename) {
  Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
  if (dataset == NULL){
    fprintf(stderr, "fail to malloc a Dataset\n");
    return NULL;
  }
  read_file(filename, dataset);
  return dataset;
}

/* Same as A2, you can reuse your code if you want! */
void free_dataset(Dataset *data) {
  for (int i = 0; i<(data->num_items); i++){
    free(data->images[i].data);
  }
  free(data->images);
  free(data->labels);
  free(data);
  return;
}


/************************** A3 Code below *************************************/

/**
 * NOTE ON AUTOTESTING:
 *    For the purposes of testing your A3 code, the actual KNN stuff doesn't
 *    really matter. We will simply be checking if (i) the number of children
 *    are being spawned correctly, and (ii) if each child is recieving the 
 *    expected parameters / input through the pipe / sending back the correct
 *    result. If your A1 code didn't work, then this is not a problem as long
 *    as your program doesn't crash because of it
 */

/**
 * This function should be called by each child process, and is where the 
 * kNN predictions happen. Along with the training and testing datasets, the
 * function also takes in 
 *    (1) File descriptor for a pipe with input coming from the parent: p_in
 *    (2) File descriptor for a pipe with output going to the parent:  p_out
 * 
 * Once this function is called, the child should do the following:
 *    - Read an integer `start_idx` from the parent (through p_in)
 *    - Read an integer `N` from the parent (through p_in)
 *    - Call `knn_predict()` on testing images `start_idx` to `start_idx+N-1`
 *    - Write an integer representing the number of correct predictions to
 *        the parent (through p_out)
 */
void child_handler(Dataset *training, Dataset *testing, int K, 
                   int p_in, int p_out) {
    // initialize variable
    int start_idx;
    int N;
    int count = 0;
    // Read an integer `start_idx` from the parent (through p_in)
    if (read(p_in, &start_idx, sizeof(int)) == -1){
      perror("reading start_idx from parent to chile");
    }
    // Read an integer `N` from the parent (through p_in)
    if (read(p_in, &N, sizeof(int)) == -1){
      perror("reading N from parent to child");
    }
    //printf("child: start_idx = %d, N = %d\n", start_idx, N);
    // Call `knn_predict()` on testing images `start_idx` to `start_idx+N-1`
    for (int i = start_idx; i < start_idx + N; i++){
      if (knn_predict(training, &(testing->images[i]), K) == testing->labels[i]){
        count++;
        //printf("%d\n", i);
      }
    }
    //printf("count = %d\n", count);
    // Write an integer representing the number of correct predictions to the parent (through p_out)
    if (write(p_out, &count, sizeof(int)) != sizeof(int)){
      perror("write from child to parent");
    }
  return;
}
#include "knn.h"

/**
 * This function takes in the name of the text file containing the image names
 * This function should NOT assume anything about the image resolutions,
 * and may be tested with images of different sizes. You will need to allocate
 * your own data for the struct, and also for the images / labels.
 */

unsigned char parse_image_path(char *image_path)
{
  int i = 0;
  char wrong = 'a';
  while (i < strlen(image_path))
  {
    if (image_path[i] == '.' && image_path[i - 2] == '-')
    {
      return atoi(&image_path[i - 1]);
    }
    i++;
  }
  return wrong;
}

void read_image_path(char *image_path, Image *image)
{
  char input_str[4];
  FILE *image_file = fopen(image_path, "r");
    if(image_file == 0) {
    perror("fopen");
    exit(1);
  }
  int index = 0;
  while ((fscanf(image_file, "%s", input_str) != EOF) && index < 4){
    if (index == 1){
      image->sx = atoi(input_str);
    }
    if (index == 2){
      image->sy = atoi(input_str);
    }
    index++;
  }
  rewind(image_file);
  index = 0;
  int size = (image->sx)*(image->sy);
  image->data = (unsigned char*)malloc(size*sizeof(unsigned char));
  while ((fscanf(image_file, "%s", input_str) != EOF)){
    if (index>3){
      image->data[index-4] = atoi(input_str);
    }
    index++;
  }
  fclose(image_file);
}

void read_file(const char *filename, Dataset *dataset)
{
  char image_path[256];
  FILE* input_file;
  input_file = fopen(filename, "r");
  if(input_file == 0) {
    perror("fopen");
    exit(1);
  }
  int count = 0;
  int i = 0;
  while (fscanf(input_file, "%s", image_path) != EOF)
  {
    count++;
  }
  dataset->num_items = count;
  dataset->images = (Image *)malloc(count * sizeof(Image));
  dataset->labels = (unsigned char *)malloc(count * sizeof(unsigned char));
  rewind(input_file);
  while (fscanf(input_file, "%s", image_path) != EOF)
  {
    dataset->labels[i] = parse_image_path(image_path);
    read_image_path(image_path, &(dataset->images[i]));
    i++;
  }
  fclose(input_file);
}

Dataset *load_dataset(const char *filename)
{
  // TODO: Allocate data, read image data / labels, return
  // initialize struct dataset
  Dataset *dataset = (Dataset *)calloc(1, sizeof(Dataset));
  if (dataset == NULL)
  {
    return NULL;
  }
  read_file(filename, dataset);
  return dataset;
}



/****************************************************************************/
/* For all the remaining functions you may assume all the images are of the */
/*     same size, you do not need to perform checks to ensure this.         */
/****************************************************************************/

/*this struct store labels and distance of an Image */
typedef struct {
  unsigned char label;
  double distance;
} KNode;

/** 
 * Return the euclidean distance between the image pixels (as vectors).
 */
double distance(Image *a, Image *b)
{
  int size = a->sx*a->sy;
  double sum_dist = 0.0;
  for (int i = 0; i < size; i++){
    int diff = ((a->data[i]) - (b->data[i]));
    sum_dist = sum_dist + diff*diff;
  }
  return sqrt(sum_dist);
}

/**
 * Given the input training dataset, an image to classify and K,
 *   (1) Find the K most similar images to `input` in the dataset
 *   (2) Return the most frequent label of these K images
 * 
 * Note: If there's multiple images with the same smallest values, pick the
 *      ones that come first. For automarking we will make sure the K smallest
 *      ones are unique, so it doesn't really matter.
 */

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

/** 
 * Free all the allocated memory for the dataset
 */
void free_dataset(Dataset *data)
{
  for (int i = 0; i<(data->num_items); i++){
    free(data->images[i].data);
  }
  free(data->images);
  free(data->labels);
  free(data);
}

/*
int main() {
    Dataset* dataset = load_dataset("my_trainning.txt");
    Dataset* testset = load_dataset("my_testing.txt");
    printf("%d\n", dataset->num_items);
    //printf("show the labels:\n");
    //for (int i = 0; i<(dataset->num_items); i++) {
    //  printf("%c ", dataset->labels[i]);
    //}
    //printf("\nshow the images:\n");
    //int i = 0;
    //while(i<3){
    //  printf("sx: %d, sy: %d\n", dataset->images[i].sx, dataset->images[i].sy);
    //  i++;
    //}
    //printf("show data:\n");
    //for (int i =0; i<(28*28); i++){
    //  printf("%d ", dataset->images[1].data[i]);
    //}
    //for (int i = 0; i<25; i++){
    //  double result = distance(&(testset->images[0]), &(testset->images[i]));
    //  printf("knn is: %f \n", result);
    //}
    for (int i = 0; i<20; i++){
      int r = knn_predict(dataset, &(testset->images[i]), 8);
      printf("knn_predict is: %d \n", r);
    }
    free_dataset(dataset);
    return 0;
}
*/

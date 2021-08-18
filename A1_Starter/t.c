#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int args, char **argv){
  char a[10];
  a[0] = 0;
  a[1] = 253;
  char p = '0';
  int c = (int) p;
  printf("%d ", c);
  printf("%d %d", a[0], a[1]);
  return 0;
}

//testing here
int main() {
    Dataset* dataset = load_dataset("lists/training_1k.txt");
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
    for (int i = 0; i<1; i++){
      int r = knn_predict(dataset, &(testset->images[i]), 10);
      printf("knn_predict is: %d \n", r);
    }
    free_dataset(dataset);
    return 0;
}

#include "knn.h"

// Makefile included in starter:
//    To compile:               make
//    To decompress dataset:    make datasets
//
// Example of running validation (K = 3, 8 processes):
//    ./classifier 3 datasets/training_data.bin datasets/testing_data.bin 8

/*****************************************************************************/
/* This file should only contain code for the parent process. Any code for   */
/*      the child process should go in `knn.c`. You've been warned!          */
/*****************************************************************************/

/**
 * main() takes in 4 command line arguments:
 *   - K:  The K value for kNN
 *   - training_data: A binary file containing training image / label data
 *   - testing_data: A binary file containing testing image / label data
 *   - num_procs: The number of processes to be used in validation
 * 
 * You need to do the following:
 *   - Parse the command line arguments, call `load_dataset()` appropriately.
 *   - Create the pipes to communicate to and from children
 *   - Fork and create children, close ends of pipes as needed
 *   - All child processes should call `child_handler()`, and exit after.
 *   - Parent distributes the testing set among childred by writing:
 *        (1) start_idx: The index of the image the child should start at
 *        (2)    N:      Number of images to process (starting at start_idx)
 *     Each child should gets N = ceil(test_set_size / num_procs) images
 *      (The last child might get fewer if the numbers don't divide perfectly)
 *   - Parent waits for children to exit, reads results through pipes and keeps
 *      the total sum.
 *   - Print out (only) one integer to stdout representing the number of test 
 *      images that were correctly classified by all children.
 *   - Free all the data allocated and exit.
 */

int main(int argc, char *argv[]) {
  // TODO: Handle command line arguments
  int K = atoi(argv[1]);
  int num_procs = atoi(argv[4]);
  Dataset* training_data = load_dataset(argv[2]);
  Dataset* testing_data = load_dataset(argv[3]);
  //
  //testing_data->num_items = 200;
  //
  int pipes[num_procs * 2][2];    // pipes[pIndex]: write from parent to child; 
                                  // pipes[pIndex + 1]: write from child to parent
  int pIndex = 0;
  int start_idx_arr[num_procs];
  int ceil_N = ceil((double)testing_data->num_items / (double)num_procs);
  int index = 0;
  for (int i = 0; i < num_procs; i++){
    start_idx_arr[i] = index;
    index = index + ceil_N;
  }
  // TODO: Spawn `num_procs` children
  for (int i = 0; i < num_procs; i++){
    if ((pipe(pipes[pIndex])) == -1){  
      perror("pipe");
      exit(1);
    }
    if ((pipe(pipes[pIndex + 1])) == -1){
      perror("pipe");
      exit(1);
    }
    int result = fork();
    if (result == -1){    // fork failed
      perror("fork");
      exit(1);
    }
    else if (result == 0){    // child case
      // close write port of pIndex
      close(pipes[pIndex][1]);
      // close read port of pIndex+1
      close(pipes[pIndex+1][0]);
      // close every read port before pIndex
      for (int process = 1; process < pIndex + 1; process += 2){
        close(pipes[process][0]);
      }
      child_handler(training_data, testing_data, K, pipes[pIndex][0], pipes[pIndex+1][1]);
      // finish writing, close the writing end of pipes[pIndex+1]
      close(pipes[pIndex+1][1]);
      close(pipes[pIndex][0]);
      // child exit
      //
      //testing_data->num_items = 10000;
      //
      free_dataset(testing_data);
      free_dataset(training_data);
      exit(0);
    }
    // TODO: Send information to children
    else{               // parent case
      // close the reading end of pIndex
      close(pipes[pIndex][0]);
      // close every read port before
      for (int process = 0; process < pIndex; process += 2){
        close(pipes[process][0]);
      }
      // close the writing end of pIndex + 1
      close(pipes[pIndex + 1][1]);
      // write to child
      int start_idx = start_idx_arr[i];
      int N;
      if (i != num_procs-1){
        N = start_idx_arr[i+1] - start_idx_arr[i];
      }
      else{
        N = testing_data->num_items - start_idx_arr[i];
      }
      //printf("parent: start_idx = %d, N = %d\n", start_idx, N);
      // write start_idx
      if (write(pipes[pIndex][1], &start_idx, sizeof(int)) < 0){
        perror("write to child");
        exit(1);
      }
      // write N
      if (write(pipes[pIndex][1], &N, sizeof(int)) < 0){
        perror("write to child");
        exit(1);
      }
      close(pipes[pIndex][1]);
    }
    // increase pIndex
    pIndex = pIndex + 2;
  }
  
  // TODO: Compute the total number of correct predictions from returned values
  int total_correct = 0;
  int part = 0;
  for (int i = 0; i< num_procs*2; i += 2){
    if (read(pipes[i+1][0], &part, sizeof(int)) < 0){
      perror("read from child");
      exit(1);
    }
    close(pipes[i+1][0]);
    //printf("parentfinal: %d\n", part);
    total_correct += part;
  }
  // Print out answer
  //
  //testing_data->num_items = 10000;
  //
  free_dataset(testing_data);
  free_dataset(training_data);
  printf("%d\n", total_correct);
  return 0;
}

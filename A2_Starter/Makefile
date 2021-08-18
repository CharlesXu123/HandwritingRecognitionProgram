CC = gcc
CFLAGS = -g -Wall -std=c99
LDLIBS = -lm

all: classifier 

# Use this to compile manually:
#  gcc -o classifier dectree.c classifier.c -g -Wall -std=c99 -lm

classifier: dectree.c classifier.c
	$(CC) -o $@ $^ $(CFLAGS) $(LDLIBS)

datasets: datasets.tgz
	tar xvzf datasets.tgz

.PHONY: clean all

clean:	
	rm -rf *.dSYM classifier

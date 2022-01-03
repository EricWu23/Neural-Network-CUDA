include sources.mk

CFLAGS =-std=c++11 $(INCLUDES:%=-I %)

CC = nvcc
LD= nvcc

OBJS1 = $(SOURCES:.cpp=.o)
OBJS = $(OBJS1:.cu=.o)
all: *.cu
	$(CC) $(CFLAGS) $(SOURCESLOCATION) --device-c
	$(LD) $(OBJS) -L /usr/local/cuda/lib -lcudart -o gpukernel
.PHONY: clean
clean:
	rm -f $(OBJS) gpukernel $(IRRELEVANT)
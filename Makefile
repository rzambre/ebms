# -*- Mode: Makefile; -*-
CC=icc
CFLAGS= -g3 -O3 -Wall -lmpi
OMPFLAGS= -fopenmp
EBMS_COMMON_SRC=ebms_common.c
BINS=ebms_single_shared_mem
BINS+=ebms_multiple
BINS+=ebms_multiple_nwins
BINS+=ebms_multiple_nwins_window_sharing

all: $(BINS)

ebms_single_shared_mem: ebms_single_shared_mem.c $(EBMS_COMMON_SRC)
	$(CC) $(CFLAGS) $^ -o $@

ebms_multiple: ebms_multiple.c $(EBMS_COMMON_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@

ebms_multiple_nwins: ebms_multiple_nwins.c $(EBMS_COMMON_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@

ebms_multiple_nwins_window_sharing: ebms_multiple_nwins_window_sharing.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@

clean:
	rm -f $(BINS)

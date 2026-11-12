# Build the Assignment 3 program. Same flags as Assignments 1-2 plus -fopenmp.
CC     := gcc
CFLAGS := -O2 -std=c17 -Wall -Wextra -ffp-contract=off -fopenmp

.PHONY: all test clean
all: build/matmul3

build/matmul3: c/matmul3.c | build
	$(CC) $(CFLAGS) -o $@ $<

build:
	mkdir -p build

test: all
	./build/matmul3 --selftest

clean:
	rm -rf build

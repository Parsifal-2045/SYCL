#!/bin/bash

python3 produce-data.py

g++ to_bin.cc
./a.out

g++ read_bin.cc
./a.out

rm ./a.out
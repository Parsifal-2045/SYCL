# alpaka 
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 1 >> alpaka_cuda_1.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 2 >> alpaka_cuda_2.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 4 >> alpaka_cuda_4.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 6 >> alpaka_cuda_6.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 8 >> alpaka_cuda_8.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 10 >> alpaka_cuda_10.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 12 >> alpaka_cuda_12.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 14 >> alpaka_cuda_14.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 16 >> alpaka_cuda_16.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 18 >> alpaka_cuda_18.txt; sleep 1; done
for i in {1..10}; do CUDA_VISIBLE_DEVICES=2 numactl -N 1 ./alpaka --dim 3 --cuda --maxEvents 10000 --numberOfThreads 20 >> alpaka_cuda_20.txt; sleep 1; done

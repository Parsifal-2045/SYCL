for i in {1..10}; do taskset -c 0 ./alpaka --syclcpu --dim 3 --maxEvents 100 --numberOfThreads 1 >> alpaka_sycl_1.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-1 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 2 >> alpaka_sycl_2.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-3 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 4 >> alpaka_sycl_4.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-5 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 6 >> alpaka_sycl_6.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-7 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 8 >> alpaka_sycl_8.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-9 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 10 >> alpaka_sycl_10.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-11 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 12 >> alpaka_sycl_12.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-13 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 14 >> alpaka_sycl_14.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-15 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 16 >> alpaka_sycl_16.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-17 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 18 >> alpaka_sycl_18.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-19 ./alpaka --syclcpu --dim 3 --maxEvents 1000 --numberOfThreads 20 >> alpaka_sycl_20.txt; sleep 1; done
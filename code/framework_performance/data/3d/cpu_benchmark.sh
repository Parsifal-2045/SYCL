#sycl
for i in {1..10}; do taskset -c 0 ./sycl --dim 3 --device cpu --maxEvents 100 --numberOfThreads 1 >> cpu_1.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-1 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 2 >> cpu_2.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-3 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 4 >> cpu_4.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-5 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 6 >> cpu_6.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-7 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 8 >> cpu_8.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-9 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 10 >> cpu_10.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-11 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 12 >> cpu_12.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-13 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 14 >> cpu_14.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-15 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 16 >> cpu_16.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-17 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 18 >> cpu_18.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-19 ./sycl --dim 3 --device cpu --maxEvents 1000 --numberOfThreads 20 >> cpu_20.txt; sleep 1; done

#alpaka and serial
for i in {1..10}; do taskset -c 0-1 ./alpaka --dim 3 --serial --maxEvents 1000 --numberOfThreads 2 >> new_alpaka_2.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-1 ./serial --dim 3 --maxEvents 1000 --numberOfThreads 2 >> new_serial_2.txt; sleep 1; done

for i in {1..10}; do taskset -c 0-3 ./alpaka --dim 3 --serial --maxEvents 1000 --numberOfThreads 4 >> new_alpaka_4.txt; sleep 1; done
for i in {1..10}; do taskset -c 0-3 ./serial --dim 3 --maxEvents 1000 --numberOfThreads 4 >> new_serial_4.txt; sleep 1; done
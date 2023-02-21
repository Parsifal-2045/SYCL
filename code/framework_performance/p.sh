# set the variables
file=sycl_intel_gpu.txt
device=level_zero
maxEvents=10000
reps=10
maxThreads=2

# run the benchmark
for ((i = 0; i <= $maxThreads; i+=2))
do
   y=$i
   if (( $i == 0))
   then
     y=$(( $y + 1 ))
   fi
   echo "# run $maxEvents events on device $device with $y threads" >> $file
   for (( j = 0; j < $reps; j++ ))
   do (numactl -N 1 ./sycl --device $device --maxEvents $maxEvents  --numberOfThreads $y 2>/dev/null | awk '/^Processed/ {printf "%.4f;",$8}' >> $file)
   done
done

# set the variables
file=alpaka_sycl_intel_gpu.txt
device=level_zero
maxEvents=10000
reps=10
maxThreads=2

# run the benchmark
for ((i = 0; i <= $maxThreads; i+=2))
do
   y=$i
   if (( $i == 0))
   then
     y=$(( $y + 1 ))
   fi
   echo "# run $maxEvents events on device $device with $y threads" >> $file
   for (( j = 0; j < $reps; j++ ))
   do (numactl -N 1 ./alpaka --syclgpu --maxEvents $maxEvents  --numberOfThreads $y 2>/dev/null | awk '/^Processed/ {printf "%.4f;",$8}' >> $file)
   done
done
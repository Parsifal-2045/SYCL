# SYCL
Testing physics implementation of SYCL and OneApi

# SYCL version of CLUE
After installing the required dependecies (OneAPI) source dpcpp local variables:
```
. /opt/intel/oneapi/setvars.sh
```
Compiling the SYCL version of CLUE only requires to go to the directory code/clue_OneAPI and run the command:
```
make mainSYCL
```
We kept the main code as similar to the original implementation as possible, so you can execute a test run of the SYCL version:
```cpp
./mainSYCL data/input/aniso_1000.csv 20 25 2 1 10 1

// In general the parameters are as follows:
./mainSYCL [fileName] [dc] [rhoc] [outlierDeltaFactor] [useGPU] [totalNumberOfEvent] [verbose]
// The parameter useGPU is used to determine wether to run the native CPU code (0) or the SYCL code (1)
```
The SYCL version of CLUE defaults to run on an Intel GPU if one isn't available, it falls back to the CPU. You may notice a way longer runtime during the first iteration of the algorithm, this is due to the runtime compilation of the code on your specific device. At this moment, the code is precompiled for only one specific GPU. Consider that with ahead-of-time compilation all of the iterations take approximately the same time, while if your specific device isn't targeted the first run has a longer execution time which accounts for device-specific compilation.
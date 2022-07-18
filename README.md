# SYCL
Testing physics implementation of SYCL and OneApi

# SYCL version of CLUE
After installing the required dependecies (OneAPI) source dpcpp local variables:
```
. /opt/intel/oneapi/setvars.sh
```
Compiling the SYCL version of CLUE only requires to run:
```
make mainSYCL
```
We kept the main code as similar to the original implementation as possible, so you can execute a test run:
```cpp
./mainSYCL data/input/aniso_1000.csv 20 25 2 0 10 1

//in general the parameters are as follows:
./mainSYCL [fileName] [dc] [rhoc] [outlierDeltaFactor] [useGPU] [totalNumberOfEvent] [verbose]
//Right now the parameter [useGPU] does nothing in this version of the code.
```
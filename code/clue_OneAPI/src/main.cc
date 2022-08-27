#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <regex>
#include "CLUEAlgo.h"
#ifndef USE_CUPLA
#ifndef USE_SYCL
#include "CLUEAlgoGPU.h"
#endif
#ifdef USE_SYCL
#include "CLUEAlgoSYCL.h"
#endif
#else
#include "CLUEAlgoCupla.h"
#ifdef FOR_TBB
#include "tbb/task_scheduler_init.h"
#endif
#endif

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

std::string create_outputfileName(std::string inputFileName, float dc,
                                  float rhoc, float outlierDeltaFactor,
                                  bool useParallel)
{
    std::string underscore = "_", suffix = "";
    suffix.append(underscore);
    suffix.append(to_string_with_precision(dc, 2));
    suffix.append(underscore);
    suffix.append(to_string_with_precision(rhoc, 2));
    suffix.append(underscore);
    suffix.append(to_string_with_precision(outlierDeltaFactor, 2));
    suffix.append(".csv");

    std::string tmpFileName;
    std::regex regexp("input");
    std::regex_replace(back_inserter(tmpFileName),
                       inputFileName.begin(), inputFileName.end(), regexp, "output");

    std::string outputFileName;
    std::regex regexp2(".csv");
    std::regex_replace(back_inserter(outputFileName),
                       tmpFileName.begin(), tmpFileName.end(), regexp2, suffix);

    return outputFileName;
}

void mainRun(std::string inputFileName, std::string outputFileName,
             float dc, float rhoc, float outlierDeltaFactor,
             bool useGPU, int repeats, bool verbose)
{
    //////////////////////////////
    // read toy data from csv file
    //////////////////////////////
    std::cout << "Start to load input points" << std::endl;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<int> layer;
    std::vector<float> weight;

    // make dummy layers
    for (int l = 0; l < NLAYERS; l++)
    {
        // open csv file
        std::ifstream iFile(inputFileName);
        std::string value = "";
        // Iterate through each line and split the content using delimeter
        while (getline(iFile, value, ','))
        {
            x.push_back(std::stof(value));
            getline(iFile, value, ',');
            y.push_back(std::stof(value));
            getline(iFile, value, ',');
            layer.push_back(std::stoi(value) + l);
            getline(iFile, value);
            weight.push_back(std::stof(value));
        }
        iFile.close();
    }
    std::cout << "Finished loading input points" << std::endl;

    //////////////////////////////
    // Make correct CLUEAlgo Hardware implementation
    //////////////////////////////
#if defined(USE_SYCL)
    std::cout << "Using CLUEAlgoSYCL: " << std::endl;
    CLUEAlgoSYCL* clueAlgo = new CLUEAlgoSYCL(dc, rhoc, outlierDeltaFactor, verbose, useGPU);
#elif defined(USE_CUPLA)
    std::cout << "Using CLUEAlgoCupla: " << std::endl;
    CLUEAlgoCupla<cupla::Acc>* clueAlgo = new CLUEAlgoCupla<cupla::Acc>(dc, rhoc, outlierDeltaFactor, verbose);
#else
    CLUEAlgo* clueAlgo;
    if (useGPU)
    {
        std::cout << "Using CLUEAlgoGPU: " << std::endl;
        clueAlgo = new CLUEAlgoGPU(dc, rhoc, outlierDeltaFactor, verbose);
    }
    else {
        std::cout << "Using CLUEAlgo: " << std::endl;
        //todo: can print cpu name by reading and prasing /proc/cpuinfo file
        // it has to be here to have a program output identical to other implementations to do 
        // automatic benchmarking measurements gathering
        std::cout << "0: Using an Intel CPU (TODO: More info)" << std::endl;
        clueAlgo = new CLUEAlgo(dc, rhoc, outlierDeltaFactor, verbose);
    }
#endif

    //////////////////////////////
    // Run CLUE algorithm
    //////////////////////////////
    for (int r = 0; r < repeats; r++)
    {
        // std::cout << "Iteration " << r << ":" <<'\n';
        // auto begin_set = std::chrono::high_resolution_clock::now();
        clueAlgo->setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]);
        // auto end_set = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_set = end_set - begin_set;
        // std::cout << "Set: " << elapsed_set.count() * 1000 << "ms\n";
        auto begin_cluster = std::chrono::high_resolution_clock::now();
        clueAlgo->makeClusters();
        auto end_cluster = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_cluster = end_cluster - begin_cluster;
        std::cout << "Make clusters: " << elapsed_cluster.count() * 1000 << "ms\n";
    }
    // output results to outputFileName. -1 means all points.
    if (verbose)
    {
        clueAlgo->verboseResults(outputFileName, -1);
    }

    std::cout << "Finished running CLUE algorithm" << std::endl;
    delete clueAlgo;
} // end of testRun()

int main(int argc, char *argv[])
{

    //////////////////////////////
    // MARK -- set algorithm parameters
    //////////////////////////////
    float dc = 20.f, rhoc = 80.f, outlierDeltaFactor = 2.f;
    bool useGPU = false;
    int totalNumberOfEvent = 10;
    bool verbose = false;

    int TBBNumberOfThread = 1;

    std::string inputFileName = argv[1];
    if (argc == 8 || argc == 9)
    {
        dc = std::stof(argv[2]);
        rhoc = std::stof(argv[3]);
        outlierDeltaFactor = std::stof(argv[4]);
        useGPU = (std::stoi(argv[5]) == 1) ? true : false;
        totalNumberOfEvent = std::stoi(argv[6]);
        verbose = (std::stoi(argv[7]) == 1) ? true : false;
        if (argc == 9)
        {
            TBBNumberOfThread = std::stoi(argv[8]);
            if (verbose)
            {
                std::cout << "Using " << TBBNumberOfThread;
                std::cout << " TBB Threads" << std::endl;
            }
        }
    }
    else
    {
        std::cout << "bin/main [fileName] [dc] [rhoc] [outlierDeltaFactor] [useGPU] [totalNumberOfEvent] [verbose] [NumTBBThreads]" << std::endl;
        return 1;
    }

#ifdef FOR_TBB
    if (verbose)
    {
        std::cout << "Setting up " << TBBNumberOfThread << " TBB Threads" << std::endl;
    }
    tbb::task_scheduler_init init(TBBNumberOfThread);
#endif

    //////////////////////////////
    // MARK -- set input and output files
    //////////////////////////////
    std::cout << "Input file: " << inputFileName << std::endl;

    std::string outputFileName = create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor, useGPU);
    std::cout << "Output file: " << outputFileName << std::endl;

    //////////////////////////////
    // MARK -- test run
    //////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    mainRun(inputFileName, outputFileName,
            dc, rhoc, outlierDeltaFactor,
            useGPU, totalNumberOfEvent, verbose);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) / 1e6;
    std::cout << "Processed " << totalNumberOfEvent << " events in " << std::scientific << time << " seconds, throughput "
              << std::defaultfloat << totalNumberOfEvent / time << " events/s" << std::endl;

    return 0;
}

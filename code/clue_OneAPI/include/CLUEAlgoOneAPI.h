#ifndef CLUEAlgoOneAPI_h
#define CLUEAlgoOneAPI_h

#include <CL/sycl.hpp>
#include "CLUEAlgo.h"
#include "LayerTilesGPU.h"
#include "Intel_gpu_selector.h"

static const int maxNSeeds = 100000;
static const int maxNFollowers = 20;
static const int localStackSizePerSeed = 20;

struct PointsPtr
{
  float *x;
  float *y;
  int *layer;
  float *weight;

  float *rho;
  float *delta;
  int *nearestHigher;
  int *clusterIndex;
  int *isSeed;
};

class CLUEAlgoOneAPI : public CLUEAlgo
{
  // inheriate from CLUEAlgo

public:
  // constructor
  CLUEAlgoOneAPI(float dc, float rhoc, float outlierDeltaFactor, bool verbose) : CLUEAlgo(dc, rhoc, outlierDeltaFactor, verbose)
  {
    init_device();
  }
  // destructor
  ~CLUEAlgoOneAPI()
  {
    free_device();
  }

  // public methods
  void makeClusters(); // overwrite base class

private:
  // private variables

  // algorithm internal variables
  PointsPtr d_points;
  LayerTilesGPU *d_hist;
  GPU::VecArray<int, maxNSeeds> *d_seeds;
  GPU::VecArray<int, maxNFollowers> *d_followers;
  sycl::queue queue = sycl::queue{Intel_gpu_selector{}};

  // private methods
  void init_device()
  {
    unsigned int reserve = 1000000;
    // input variables
    d_points.x = sycl::malloc_device<float>(reserve, queue);
    d_points.y = sycl::malloc_device<float>(reserve, queue);
    d_points.layer = sycl::malloc_device<int>(reserve, queue);
    d_points.weight = sycl::malloc_device<float>(reserve, queue);
    // result variables
    d_points.rho = sycl::malloc_device<float>(reserve, queue);
    d_points.delta = sycl::malloc_device<float>(reserve, queue);
    d_points.nearestHigher = sycl::malloc_device<int>(reserve, queue);
    d_points.clusterIndex = sycl::malloc_device<int>(reserve, queue);
    d_points.isSeed = sycl::malloc_device<int>(reserve, queue);
    // algorithm internal variables
    d_hist = sycl::malloc_device<LayerTilesGPU>(NLAYERS, queue);
    d_seeds = sycl::malloc_device<GPU::VecArray<int, maxNSeeds>>(1, queue);
    d_followers = sycl::malloc_device<GPU::VecArray<int, maxNFollowers>>(reserve, queue);
  }

  void free_device()
  {
    // input variables
    sycl::free(d_points.x, queue);
    sycl::free(d_points.y, queue);
    sycl::free(d_points.layer, queue);
    sycl::free(d_points.weight, queue);
    // result variables
    sycl::free(d_points.rho, queue);
    sycl::free(d_points.delta, queue);
    sycl::free(d_points.nearestHigher, queue);
    sycl::free(d_points.clusterIndex, queue);
    sycl::free(d_points.isSeed, queue);
    // algorithm internal variables
    sycl::free(d_hist, queue);
    sycl::free(d_seeds, queue);
    sycl::free(d_followers, queue);
  }

  void copy_todevice()
  {
    // input variables
    queue.memcpy(d_points.x, points_.x.data(), sizeof(float) * points_.n);
    queue.memcpy(d_points.y, points_.y.data(), sizeof(float) * points_.n);
    queue.memcpy(d_points.layer, points_.layer.data(), sizeof(int) * points_.n);
    queue.memcpy(d_points.weight, points_.weight.data(), sizeof(float) * points_.n).wait();
  }

  void clear_set()
  {
    // result variables
    queue.memset(d_points.rho, 0x00, sizeof(float) * points_.n);
    queue.memset(d_points.delta, 0x00, sizeof(float) * points_.n);
    queue.memset(d_points.nearestHigher, 0x00, sizeof(int) * points_.n);
    queue.memset(d_points.clusterIndex, 0x00, sizeof(int) * points_.n);
    queue.memset(d_points.isSeed, 0x00, sizeof(int) * points_.n);
    // algorithm internal variables
    queue.memset(d_hist, 0x00, sizeof(LayerTilesGPU) * NLAYERS);
    queue.memset(d_seeds, 0x00, sizeof(GPU::VecArray<int, maxNSeeds>));
    queue.memset(d_followers, 0x00, sizeof(GPU::VecArray<int, maxNFollowers>) * points_.n);
  }

  void copy_tohost()
  {
    // result variables
    queue.memcpy(points_.clusterIndex.data(), d_points.clusterIndex, sizeof(int) * points_.n).wait();
    
    if (verbose_) // other variables, copy only when verbose_==True
    {
      queue.memcpy(points_.rho.data(), d_points.rho, sizeof(float) * points_.n);
      queue.memcpy(points_.delta.data(), d_points.delta, sizeof(float) * points_.n);
      queue.memcpy(points_.nearestHigher.data(), d_points.nearestHigher, sizeof(int) * points_.n);
      queue.memcpy(points_.isSeed.data(), d_points.isSeed, sizeof(int) * points_.n).wait();
    }
  }

};

#endif

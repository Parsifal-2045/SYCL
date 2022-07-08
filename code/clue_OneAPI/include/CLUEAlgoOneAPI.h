#ifndef CLUEAlgoOneAPI_h
#define CLUEAlgoOneAPI_h

#include <CL/sycl.hpp>
#include "CLUEAlgo.h"
#include "LayerTilesOneAPI.h"
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
  // inheritate from CLUEAlgo

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
  LayerTilesOneAPI *d_hist;
  GPU::VecArray<int, maxNSeeds> *d_seeds;
  GPU::VecArray<int, maxNFollowers> *d_followers;
  sycl::queue queue_ = sycl::queue{Intel_gpu_selector{}};

  // private methods
  void init_device()
  {
    unsigned int reserve = 1000000;
    // input variables
    d_points.x = sycl::malloc_shared<float>(reserve, queue_);
    d_points.y = sycl::malloc_shared<float>(reserve, queue_);
    d_points.layer = sycl::malloc_shared<int>(reserve, queue_);
    d_points.weight = sycl::malloc_shared<float>(reserve, queue_);
    // result variables
    d_points.rho = sycl::malloc_shared<float>(reserve, queue_);
    d_points.delta = sycl::malloc_shared<float>(reserve, queue_);
    d_points.nearestHigher = sycl::malloc_shared<int>(reserve, queue_);
    d_points.clusterIndex = sycl::malloc_shared<int>(reserve, queue_);
    d_points.isSeed = sycl::malloc_shared<int>(reserve, queue_);
    // algorithm internal variables
    d_hist = sycl::malloc_shared<LayerTilesOneAPI>(NLAYERS, queue_);
    d_seeds = sycl::malloc_shared<GPU::VecArray<int, maxNSeeds>>(1, queue_);
    d_followers = sycl::malloc_shared<GPU::VecArray<int, maxNFollowers>>(reserve, queue_);
  }

  void free_device()
  {
    // input variables
    sycl::free(d_points.x, queue_);
    sycl::free(d_points.y, queue_);
    sycl::free(d_points.layer, queue_);
    sycl::free(d_points.weight, queue_);
    // result variables
    sycl::free(d_points.rho, queue_);
    sycl::free(d_points.delta, queue_);
    sycl::free(d_points.nearestHigher, queue_);
    sycl::free(d_points.clusterIndex, queue_);
    sycl::free(d_points.isSeed, queue_);
    // algorithm internal variables
    sycl::free(d_hist, queue_);
    sycl::free(d_seeds, queue_);
    sycl::free(d_followers, queue_);
  }

  void copy_todevice()
  {
    // input variables
    queue_.memcpy(d_points.x, points_.x.data(), sizeof(float) * points_.n);
    queue_.memcpy(d_points.y, points_.y.data(), sizeof(float) * points_.n);
    queue_.memcpy(d_points.layer, points_.layer.data(), sizeof(int) * points_.n);
    queue_.memcpy(d_points.weight, points_.weight.data(), sizeof(float) * points_.n).wait();
  }

  void clear_set()
  {
    // result variables
    queue_.memset(d_points.rho, 0x00, sizeof(float) * points_.n);
    queue_.memset(d_points.delta, 0x00, sizeof(float) * points_.n);
    queue_.memset(d_points.nearestHigher, 0x00, sizeof(int) * points_.n);
    queue_.memset(d_points.clusterIndex, 0x00, sizeof(int) * points_.n);
    queue_.memset(d_points.isSeed, 0x00, sizeof(int) * points_.n);
    // algorithm internal variables
    queue_.memset(d_hist, 0x00, sizeof(LayerTilesOneAPI) * NLAYERS);
    queue_.memset(d_seeds, 0x00, sizeof(GPU::VecArray<int, maxNSeeds>));
    queue_.memset(d_followers, 0x00, sizeof(GPU::VecArray<int, maxNFollowers>) * points_.n);
  }

  void copy_tohost()
  {
    // result variables
    queue_.memcpy(points_.clusterIndex.data(), d_points.clusterIndex, sizeof(int) * points_.n).wait();
    
    if (verbose_) // other variables, copy only when verbose_==True
    {
      queue_.memcpy(points_.rho.data(), d_points.rho, sizeof(float) * points_.n);
      queue_.memcpy(points_.delta.data(), d_points.delta, sizeof(float) * points_.n);
      queue_.memcpy(points_.nearestHigher.data(), d_points.nearestHigher, sizeof(int) * points_.n);
      queue_.memcpy(points_.isSeed.data(), d_points.isSeed, sizeof(int) * points_.n).wait();
    }
  }

};

#endif

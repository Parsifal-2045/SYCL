#ifndef PointsPtr_h
#define PointsPtr_h

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

#endif
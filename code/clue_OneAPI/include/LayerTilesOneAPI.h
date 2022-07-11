#ifndef LayerTilesOneAPI_h
#define LayerTilesOneAPI_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
// OneAPI include
#include <CL/sycl.hpp>

#include "OneAPIVecArray.h"
#include "LayerTilesConstants.h"

class LayerTilesOneAPI
{

public:
  // constructor
  LayerTilesOneAPI(){};

  void fill(float x, float y, int i)
  {
    layerTiles_[getGlobalBin(x, y)].push_back(i);
  }

  int getXBin(float x) const
  {
    int xBin = (x - LayerTilesConstants::minX) * LayerTilesConstants::rX;
    xBin = (xBin < LayerTilesConstants::nColumns ? xBin : LayerTilesConstants::nColumns - 1);
    xBin = (xBin > 0 ? xBin : 0);
    return xBin;
  }

  int getYBin(float y) const
  {
    int yBin = (y - LayerTilesConstants::minY) * LayerTilesConstants::rY;
    yBin = (yBin < LayerTilesConstants::nRows ? yBin : LayerTilesConstants::nRows - 1);
    yBin = (yBin > 0 ? yBin : 0);
    return yBin;
  }

  int getGlobalBin(float x, float y) const
  {
    return getXBin(x) + getYBin(y) * LayerTilesConstants::nColumns;
  }

  int getGlobalBinByBin(int xBin, int yBin) const
  {
    return xBin + yBin * LayerTilesConstants::nColumns;
  }

  sycl::int4 searchBox(float xMin, float xMax, float yMin, float yMax)
  {
    return sycl::int4{getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
  }

  void clear()
  {
    for (auto &t : layerTiles_)
      t.reset();
  }

  OneAPI::VecArray<int, LayerTilesConstants::maxTileDepth> &operator[](int globalBinId)
  {
    return layerTiles_[globalBinId];
  }

private:
  OneAPI::VecArray<OneAPI::VecArray<int, LayerTilesConstants::maxTileDepth>, LayerTilesConstants::nColumns * LayerTilesConstants::nRows> layerTiles_;
};
#endif

#include <iostream>
#include <fstream>
#include <vector>

struct PointsCloud
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> layer;
  std::vector<float> weight;
};

struct Point
{
  float x;
  float y;
  float layer;
  float weight;
};

int main()
{
  std::ifstream in("2_events.bin", std::fstream::binary);
  uint32_t num_points;
  int j = 0;
  std::vector<PointsCloud> cloud_;

  while (true)
  {
    in.read(reinterpret_cast<char *>(&num_points), sizeof(uint32_t));
    ++j;
    if (in.eof())
    {
      break;
    }
    std::cout << "number of points event " << j << " " << num_points << std::endl;
    Point point;
    PointsCloud cloud;
    for (unsigned int i = 0; i != num_points; i++)
    {
      in.read(reinterpret_cast<char *>(&point), sizeof(Point));
      cloud.x.emplace_back(point.x);
      cloud.y.emplace_back(point.y);
      cloud.layer.emplace_back(point.layer);
      cloud.weight.emplace_back(point.weight);
    }
    cloud_.emplace_back(cloud);
    // next event
  }
  std::cout << "Processed two events" << std::endl;
  for (int i = 0; i != 10; i++)
  {
    std::cout << cloud_[0].x[i] << ", " << cloud_[0].y[i] << ", " << cloud_[0].layer[i] << ", " << cloud_[0].weight[i]
              << ", " << std::endl;
  }
  std::cout << std::endl;
  for (int i = 0; i != 10; i++)
  {
    std::cout << cloud_[1].x[i] << ", " << cloud_[1].y[i] << ", " << cloud_[1].layer[i] << ", " << cloud_[1].weight[i]
              << ", " << std::endl;
  }
}
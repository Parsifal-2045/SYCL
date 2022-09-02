#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

struct Test
{
  float x;
  float y;
  float layer;
  float weight;
};

struct PointsCloud
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> layer;
  std::vector<float> weight;
};

int main()
{
  std::ifstream infile("test_data/test_100_layers.bin", std::ios::binary);
  Test test;
  PointsCloud cloud;
  auto start = std::chrono::high_resolution_clock::now();
  while (true)
  {
    Test test;
    infile.read((char *)&test, sizeof(Test));
    if (infile.eof())
      break;
    cloud.x.emplace_back(test.x);
    cloud.y.emplace_back(test.y);
    cloud.layer.emplace_back(test.layer);
    cloud.weight.emplace_back(test.weight);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Emplace time (binary): " << elapsed.count() * 1000 << "ms\n";
  infile.close();

  // open csv file
  PointsCloud cloud2;
  infile.open("test_data/test_100_layers.csv");
  std::string value = "";
  start = std::chrono::high_resolution_clock::now();
  while (getline(infile, value, ','))
  {
    cloud2.x.emplace_back(std::stof(value));
    getline(infile, value, ',');
    cloud2.y.emplace_back(std::stof(value));
    getline(infile, value, ',');
    cloud2.layer.emplace_back(std::stof(value));
    getline(infile, value);
    cloud2.weight.emplace_back(std::stof(value));
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Emplace time (csv): " << elapsed.count() * 1000 << "ms\n";

  infile.close();

  assert(cloud.x.size() == cloud2.x.size());
  assert(cloud.y.size() == cloud2.y.size());
  assert(cloud.layer.size() == cloud2.layer.size());
  assert(cloud.weight.size() == cloud2.weight.size());
  for (int i = 0; i != cloud.x.size(); i++)
  {
    assert(cloud.x[i] == cloud2.x[i]);
    assert(cloud.y[i] == cloud2.y[i]);
    assert(cloud.layer[i] == cloud2.layer[i]);
    assert(cloud.weight[i] == cloud2.weight[i]);
  }
}
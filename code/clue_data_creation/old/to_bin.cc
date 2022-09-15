#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main()
{
  std::fstream in("test_data/test_100_layers.csv");
  std::ofstream out("test_data/test_100_layers.bin", std::ios::binary);
  float v;

  std::string line, value;
  while (std::getline(in, line))
  {
    std::istringstream l(line);
    while (std::getline(l, value, ','))
    {
      float v = std::stof(value);
      out.write(reinterpret_cast<char *>(&v), sizeof(float));
    }
  }
}
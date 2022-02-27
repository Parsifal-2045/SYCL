#include <iostream>

int main()
{
    std::cout << "Hello world!" << '\n';
    return 0;
}

// Device code

#include <iostream>

__global__ void mykernel(void)  {} // Device function -> Gets called by CPU and exectued by GPU

int main() // Host function
{
    mykernel<<<1,1>>>(); // Kernel launch -> Operator <<<a,b>>>
    std::cout << "Hello world!" << '\n';
    return 0;
}
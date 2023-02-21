#include <iostream>
#include <cassert>
#include <chrono>

template <typename T>
class Vector
{
public:
    Vector()
    {
        ReAlloc(2);
    }
    ~Vector()
    {
        delete[] data_;
    }

    void PushBack(T const &value)
    {
        if (size_ >= capacity_)
        {
            ReAlloc(capacity_ + capacity_ / 2); // memory allocation size grows geometrically
        }
        data_[size_] = value;
        size_++;
    }

    void EmplaceBack(T &&value)
    {
        if (size_ >= capacity_)
        {
            ReAlloc(capacity_ + capacity_ / 2); // memory allocation size grows geometrically
        }
        data_[size_] = T(value);
        size_++;
    }

    T &operator[](size_t index)
    {
        assert(index <= size_);
        return data_[index];
    }

    const T &operator[](size_t index) const
    {
        assert(index <= size_);
        return data_[index];
    }

    size_t Size() const
    {
        return size_;
    }

private:
    void ReAlloc(size_t new_capacity)
    {
        // 1. allocate a new block of memory
        T *new_block = new T[new_capacity];
        if (new_capacity < size_)
        {
            size_ = new_capacity;
        }
        // 2. move all of the old elements in the new block of memory
        for (size_t i = 0; i < size_; i++)
        {
            new_block[i] = std::move(data_[i]); // no memcpy, need to call the move constructor for non-primitive types
        }
        // 3. deallocate the old block of memory
        delete[] data_;
        data_ = new_block;
        capacity_ = new_capacity;
    }

    T *data_ = nullptr;
    size_t size_ = 0;     // number of elements inside the vector
    size_t capacity_ = 0; // allocated memory -> allocate more than needed to avoid reallocating at each pushback
};

template <typename T>
void PrintVector(Vector<T> const &vector)
{
    for (size_t i = 0; i < vector.Size(); i++)
    {
        std::cout << vector[i] << std::endl;
    }
    std::cout << "-----------------------------------------\n";
}

int main()
{
    auto begin = std::chrono::high_resolution_clock::now();
    Vector<std::string> vector;
    vector.PushBack("Hello");
    vector.PushBack("World");
    vector.PushBack("!");
    vector.EmplaceBack("test");
    PrintVector(vector);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = end - begin;
    auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(diff).count()) / 1e6;
    std::cout << "Time taken: " << time << std::endl;
}
# Advanced algorithms and programming methods - 2 [CM0470] - Prof. A. Torsello

## Group:

- Baccega Sandro (865024)

---

## BUILDING AND RUNNING

```bash
mkdir build
cd build
cmake ../
make
./tensor
./einstein
```

---

## USAGE AND DESIGN

We made some changes to a Tensor Library that used Einstein's Notation by making it's operations distributed over multiple threads. This results in better performances when working on very large tensors. 

We introduced the `parallel` method to use instead of the assignment operator, for example:

```cpp
tensor<size_t, rank<2>> t1(TENSOR_SIZE, TENSOR_SIZE), t2(TENSOR_SIZE, TENSOR_SIZE);

auto i = new_index;
auto j = new_index;

t2(j, i).parallel(t1(i, j));
// Gives the same result as:
// t2(j, i) = t1(i, j);
```

We used the `std::thread::hardware_concurrency()` function to get the default number of concurrent threads.

We are using 2 pools of executors: 
- `operation_executors_pool`: contains a vector of threads that performs the expression operation.  
- `cleaner_executors_pool`: contains a vector of threads that performs the clean operation (a cleaner executor fills a range with zeroes).

We used a write_mutex to prevent race conditions on the write operation in the destination tensor.

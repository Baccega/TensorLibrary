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

We made some changes to a Tensor Library that used Einstein's Notation by making it's operations distributed over multiple threads. This results in better performances when working on very large tensors. 

We introduced the `parallel` method to use instead of the assignment operator, for example:

```cpp
tensor<size_t, rank<2>> t1(TENSOR_SIZE, TENSOR_SIZE), t2(TENSOR_SIZE, TENSOR_SIZE);

auto i = index;
auto j = index;

t2(j, i).parallel(t1(i, j));
// Gives the same result as:
// t2(j, i) = t1(i, j);
```

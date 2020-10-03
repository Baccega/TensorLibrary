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
```

---

We made some changes to a Tensor Library that used Einstein's Notation by making it's operations distributed over multiple threads. This results in better performances when working on very large tensors. 

We kept the the same API from the base library, meaning that we only changed the internal behaviour of the einstein expressions assignment operator.

```cpp
tensor<size_t, rank<2>> t1(SIZE, SIZE), t2(SIZE, SIZE);

auto i = index;
auto j = index;
t2(j, i) = t1(i, j);
```

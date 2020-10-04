#include <iostream>

#include "tensor.h"

using namespace Tensor;

std::ostream &operator<<(std::ostream &out, Index_Set<>) { return out; }
template <unsigned id, unsigned... ids>
std::ostream &operator<<(std::ostream &out, Index_Set<id, ids...>) {
  return out << id << ' ' << Index_Set<ids...>();
}

#define SIZE 300

int main() {
  tensor<size_t, rank<2>> t1(SIZE, SIZE), t2(SIZE, SIZE);

  size_t count = 0;
  for (auto iter = t1.begin(); iter != t1.end(); ++iter) *iter = count++;

  auto i = new_index;
  auto j = new_index;
  auto start_time = std::chrono::high_resolution_clock::now();

  t2(j, i) = t1(i, j);

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed_time =
      std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "elapsed time: " << elapsed_time << '\n';

  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';
  start_time = std::chrono::high_resolution_clock::now();

  t2(j, i).parallel(t1(i, j));

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "parallel elapsed time: " << elapsed_time << '\n';

  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  tensor<size_t> t3(SIZE, SIZE, SIZE), t4(SIZE);
  auto k = new_index;
  count = 0;
  for (auto iter = t3.begin(); iter != t3.end(); ++iter) *iter = count++;
  start_time = std::chrono::high_resolution_clock::now();

  t4(i) = t3(i, j, k) * t1(j, k) + t3(i, k, k);

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "elapsed time: " << elapsed_time << '\n';

  for (auto iter = t4.begin(); iter != t4.end(); ++iter)
  std::cout << *iter << ' ';
  std::cout << '\n';
  start_time = std::chrono::high_resolution_clock::now();

  t4(i).parallel(t3(i, j, k) * t1(j, k) + t3(i, k, k));

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "parallel elapsed time: " << elapsed_time << '\n';

  for (auto iter = t4.begin(); iter != t4.end(); ++iter)
  std::cout << *iter << ' ';
  std::cout << '\n';
  start_time = std::chrono::high_resolution_clock::now();

  t2(i, j) = t1(i, k) * t1(k, j);

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "elapsed time: " << elapsed_time << '\n';

  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';
  start_time = std::chrono::high_resolution_clock::now();

  t2(i, j).parallel(t1(i, k) * t1(k, j));

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "parallel elapsed time: " << elapsed_time << '\n';

  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';
  start_time = std::chrono::high_resolution_clock::now();

  t2(i, k) = t3(i, j, j) * t4(k);

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "elapsed time: " << elapsed_time << '\n';

  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';
  start_time = std::chrono::high_resolution_clock::now();

  t2(i, k).parallel(t3(i, j, j) * t4(k));

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
  std::cerr << "parallel elapsed time: " << elapsed_time << '\n';

  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  std::cout << "DONE";

  return 0;
}

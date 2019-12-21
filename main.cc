#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include "Tensor.hpp"

// --- Utils
// fills tensor with index values
template <typename T>

void fill_tensor_i(T &tensor, int i = 0, int m = 1) {
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    *it = i * m;

    i++;
  }
}

// fills tensor with a constant c
template <typename T>

void fill_tensor_c(T &tensor, int c = 1) {
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    *it = c;
  }
}

template <class TensorType>
void printTensor(const TensorType &t, std::ostream &stream = std::cout) {
  for (auto iterator = t.begin(); iterator != t.end(); iterator++) {
    stream << *iterator << ", ";
  }

  stream << std::endl;
}

template <class TensorType>
void assertTensorValues(const TensorType &tensor, std::string expected) {
  std::stringstream buffer;

  printTensor(tensor, buffer);

  assert(buffer.str().compare(expected) == 0);
}

// --- Unit Tests

void tensorMultiplicationTests() {
  std::cout << "Tensor Multiplication Tests:" << std::endl;

  tensor::tensor<int> a(4, 3, 4);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(3);

  fill_tensor_i(b, 0);

  auto exp = a["ijk"] * b["j"];

  tensor::tensor<int> c = exp.evaluate();

  printTensor(c);

  assertTensorValues(c,

                     "20, 23, 26, 29, 56, 59, 62, 65, 92, 95, 98, 101, 128, "

                     "131, 134, 137, \n");

  std::cout << std::endl;

  tensor::tensor<int> a1(4);

  fill_tensor_i(a1, 0);

  tensor::tensor<int> b1(4);

  fill_tensor_i(b1, 0);

  auto exp2 = a1["i"] * b1["i"];

  tensor::tensor<int> c1 = exp2.evaluate();

  printTensor(c1);

  assertTensorValues(c1, "14, \n");

  std::cout << std::endl;

  tensor::tensor<int> a2(4, 4);

  fill_tensor_i(a2, 0);

  tensor::tensor<int> b2(4, 4);

  fill_tensor_i(b2, 0);

  tensor::tensor<int> c2(4);

  fill_tensor_i(c2, 0);

  auto exp3 = a2["ij"] * b2["ik"];

  tensor::tensor<int> d2 = exp3.evaluate();

  printTensor(d2);
}

void sameTensorMultiplicationTests() {
  std::cout << "Same Tensor Multiplication Test:" << std::endl;

  tensor::tensor<int> a(3);

  fill_tensor_i(a, 0);

  auto exp = a["i"] * a["i"];

  tensor::tensor<int> c = exp.evaluate();

  printTensor(c);

  assertTensorValues(c, "5, \n");

  tensor::tensor<int> d(3, 3);

  fill_tensor_i(d, 0);

  auto exp2 = d["ij"] * d["ij"];

  tensor::tensor<int> e = exp2.evaluate();

  printTensor(e);

  assertTensorValues(e, "204, \n");

  std::cout << std::endl;
}

void traceTests() {
  std::cout << "Trace Tests:" << std::endl;

  tensor::tensor<int> f(3, 3);

  fill_tensor_i(f, 0);

  auto exp3 = f["ii"];

  tensor::tensor<int> g = exp3.evaluate();

  printTensor(g);

  assertTensorValues(g, "12, \n");

  tensor::tensor<int> h(3, 3, 3);

  fill_tensor_i(h, 0);

  auto exp4 = h["iii"];

  tensor::tensor<int> i = exp4.evaluate();

  printTensor(i);

  assertTensorValues(i, "39, \n");

  std::cout << std::endl;
}

void tensorAdditionTests() {
  std::cout << "Tensor Addition Tests:" << std::endl;

  tensor::tensor<int> a(4);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(4);

  fill_tensor_i(b, 0);

  auto exp = a["i"] + b["i"];

  tensor::tensor<int> c = exp.evaluate();

  printTensor(c);

  assertTensorValues(c, "12, \n");

  tensor::tensor<int> a1(2, 2, 2);

  fill_tensor_i(a1, 0);

  tensor::tensor<int> b1(2, 2, 2);

  fill_tensor_i(b1, 0);

  auto exp1 = a1["ijk"] + b1["ijk"];

  tensor::tensor<int> c1 = exp1.evaluate();

  printTensor(c1);

  assertTensorValues(c1, "56, \n");

  std::cout << std::endl;
}

void tensorSubtractionTests(){
  std::cout << "Tensor Subtraction Tests:" << std::endl;

  tensor::tensor<int> a(4);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(4);

  fill_tensor_i(b, 0);

  auto exp = a["i"] - b["i"];

  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);

  tensor::tensor<int> a1(2, 2, 2);

  fill_tensor_i(a1, 0);

  tensor::tensor<int> b1(2, 2, 2);

  fill_tensor_i(b1, 0);

  auto exp1 = a1["ijk"] - b1["ijk"];

  tensor::tensor<int> c1 = exp1.evaluate();

  printTensor(c1);

  assertTensorValues(c1, "0, \n");

  std::cout << std::endl;

}

void operationConcatTests() {
  std::cout << "Tensor Operation Concat Tests:" << std::endl;

  tensor::tensor<int> a(2, 2);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(2, 2);

  fill_tensor_i(b, 0);

  tensor::tensor<int> c(2);

  fill_tensor_i(c, 0);

  auto exp = a["ij"] * b["ik"] * c["j"];

  tensor::tensor<int> d = exp.evaluate();

  printTensor(d);

  // tensor::tensor<int> a(4);

  // fill_tensor_i(a, 0);

  // tensor::tensor<int> b(4);

  // fill_tensor_i(b, 0);

  // tensor::tensor<int> c(4);

  // fill_tensor_i(c, 0);

  // auto exp = a["i"] * b["i"] * c["i"];

  // tensor::tensor<int> d = exp.evaluate();

  // printTensor(d);

  // assertTensorValues(d, "12, \n");

  // tensor::tensor<int> a1(2,2,2);

  // fill_tensor_i(a1, 0);

  // tensor::tensor<int> b1(2,2,2);

  // fill_tensor_i(b1, 0);

  // auto exp1 = a1["ijk"] + b1["ijk"];

  // tensor::tensor<int> c1 = exp1.evaluate();

  // printTensor(c1);

  // assertTensorValues(c1, "56, \n");

  std::cout << std::endl;
}

void mixedConcatTests() {
  std::cout << "Tensor Mixed Operation Concat Tests:" << std::endl;

  tensor::tensor<int> a(2, 2);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(2, 2);

  fill_tensor_i(b, 0);

  tensor::tensor<int> c(2, 2);

  fill_tensor_c(c, 1000);

  auto exp = a["ij"] * b["jk"] + c["kl"];

  tensor::tensor<int> d = exp.evaluate();

  printTensor(d);

  std::cout << std::endl;
}

void tensorConcatenation() {
  std::cout << "Concatenatation example:" << std::endl;

  tensor::tensor<int> a(3);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(3);

  fill_tensor_i(b, 0, 100);

  auto exp = a["i"] + b["j"];

  tensor::tensor<int> d = exp.evaluate();

  printTensor(d);

  std::cout << "The rank of the new tensor is " << d.get_rank() << std::endl
            << std::endl;
}

void KroneckerProduct() {
  std::cout << "Kronecker example from wikipedia:" << std::endl;

  tensor::tensor<int> a(2, 2);

  a(0, 0) = 1;

  a(0, 1) = 2;

  a(1, 0) = 3;

  a(1, 1) = 1;

  tensor::tensor<int> b(2, 2);

  b(0, 0) = 0;

  b(0, 1) = 3;

  b(1, 0) = 2;

  b(1, 1) = 1;

  auto exp = a["ij"] * b["kl"];

  tensor::tensor<int> d = exp.evaluate();  //.flatten(0,2);

  printTensor(d);

  std::cout << "The rank of the new tensor is " << d.get_rank() << std::endl;

  std::cout
      << "The operation returns a tensor of rank 2*2, but the elements are "
         "multiplied correctly.\n"

      << "If there wasn't a bug in the flatten operation one could easily "
         "implement a Kronecker product exploiting this functionality"

      << std::endl
      << std::endl;
}

void numberOfIndexesCheck() {
  std::cout << "This fails an assert (mismatch between number of indexes and "
               "dimensions)"
            << std::endl;

  tensor::tensor<int> a(2, 2);

  fill_tensor_i(a, 0);

  tensor::tensor<int> b(2);

  fill_tensor_i(b, 0);

  auto exp = a["ij"] * b["jk"];

  tensor::tensor<int> d = exp.evaluate();

  printTensor(d);

  std::cout << "Shouldn't work. Nothing gets printed." << std::endl;

  std::cout << std::endl;
}

void invalidTrace() {
  std::cout
      << "This fails an assert (dimensions of different size have same index)"
      << std::endl;

  tensor::tensor<int> a(2, 3);

  fill_tensor_i(a, 0);

  auto exp = a["ii"];

  auto d = exp.evaluate();

  printTensor(d);

  std::cout << "Shouldn't work dynamically." << std::endl;

  std::cout << std::endl;
}

void breaksComputation() {
  std::cout << "This also should break the code" << std::endl;

  auto a = tensor::tensor<int>(3, 4);

  fill_tensor_i(a, 0);

  auto b = tensor::tensor<int>(2, 2);

  fill_tensor_i(b, 0);

  auto exp2 = a["ij"] * b["jk"];

  auto d = exp2.evaluate();

  printTensor(d);

  std::cout << "Shouldn't work at all." << std::endl << std::endl;

  std::cout << "A combination of contraction and other operations with shared "
               "free indexes break the program"
            << std::endl;

  a = tensor::tensor<int>(2, 2);

  b = tensor::tensor<int>(2, 2);

  fill_tensor_i(a, 0);

  fill_tensor_i(b, 0); 

  auto exp3 = a["ii"] + b["ij"];

  auto res = exp3.evaluate();

  printTensor(res);

  std::cout << std::endl;
}



// --- Main

int main() {
  tensorMultiplicationTests();

  sameTensorMultiplicationTests();

  traceTests();

  tensorAdditionTests();

  tensorSubtractionTests();

  operationConcatTests();

  mixedConcatTests();

  tensorConcatenation();

  KroneckerProduct();

  breaksComputation();

  return 0;
}
Sandro Baccega(865024)

Arnaldo Santoro (822274)

Due date: 23/12/2019

# TODO:

- Finire Docs
- Cambiare interfaccia (a.ein("i") => a("i"))
- Migliorare i controlli a compile time?

## REQUIREMENT SPECIFICATION

### Assignment 2: Einstein's notation

Augment the tensor library with the ability to perform tensor operations using Einstein's notation.

Einstein's notation, introduced in 1916 to simplify the notation used for general relativity, expresses tensor operations by rendering implicit the summation operations. According to this notation, repeated indexes in a tensorial expression are implicitly summed over, so the expression

a_ijk b_j

represents a rank 2 tensor c indexed by i and k such that

c_ik = Σj a_ijk b_j

The notation allows for simple contractions

Tr(a) = a_ii

as well as additions (subtractions) and multiplications.

The operations should work with dynamically ranked tensors as well as fixed-rank tensors by maintaining and checking as much static information as possible.

**Hint**: define index types. indexing operations with index types should return proxy objects that allow deferred operations and conversion to the correct tensor type.

## USAGE

### Examples

#### Trace

#### Multiplication

#### Conversion

#### Combinations

#### Generalized Kronecker Product

The operation OP between two tensors A and B or rank r and s with no common indexes returns a tensor of rank r\*s whose ranks' elements are the result of OP between each element of A and each element of the rank of B .

This results similar to a Kronecker product which stores the results in different ranks of a tensor.

If there wasn't a bug in the flatten operation one could easily implement a Kronecker product exploiting this functionality.

## DESIGN

### Composite Pattern

We thought of the Composite Pattern to implement tensor operations in a uniform way.

This is obtained through different implementation of the function `evaluate()`, which behaves differently for each class implementing the operations.

### Strategy Pattern

A definition of the _strategy pattern_ is a reusable solution that lets an algorithm vary independently from clients that use it; in our code we use this solution to implement the different operations in the implicit summation of Einstein's formalism; this application is combined with the template system to obtain a _static strategy pattern_ implementation.

The object `tensor_op` is forward-declared and consists of the _functor_ applying the algorithm.

The following `struct`s are "functors" (in a broader sense) implementing the operations between two elements of type T, which are then placed in the specialized template of the definition for the `operator+`, and `operator*` respectively.

## Bugs

A combination of contraction and other operations with shared free indexes break the program.

This happens because the tensors are evaluated right out when they are passed to objects.(...)

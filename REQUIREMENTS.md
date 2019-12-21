# REQUIREMENT SPECIFICATION

### Assignment 2: Einstein's notation

### Due date: 23/12/2019

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

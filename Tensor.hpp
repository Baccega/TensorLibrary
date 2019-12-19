#ifndef TENSOR
#define TENSOR

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::rel_ops;

namespace tensor
{

// policy for dynamically ranked tensors
struct dynamic
{
  typedef std::vector<size_t> index_type;
  typedef std::vector<size_t> width_type;
};

// policy for fixed-rank tensors
template <size_t R>
struct rank
{
  typedef std::array<size_t, R> index_type;
  typedef std::array<size_t, R> width_type;
};

// tensor type
template <typename T, class type = dynamic>
class tensor;

namespace reserved
{

// generic iterator used by all tensor classes (except rank 1 specializations)
template <typename T, class type>
class iterator
{
public:
  T &operator*() const { return *ptr; }

  iterator &operator++()
  {
    // I am using a right-major layout
    // start increasing the last index
    size_t index = stride.size() - 1;
    ++idx[index];
    ptr += stride[index];
    // as long as the current index has reached maximum width,
    // set it to 0 and increase the next index
    while (idx[index] == width[index] && index > 0)
    {
      idx[index] = 0;
      ptr -= width[index] * stride[index];
      --index;
      ++idx[index];
      ptr += stride[index];
    }
    return *this;
  }

  iterator operator++(int)
  {
    iterator result(*this);
    operator++();
    return result;
  }
  iterator &operator--()
  {
    // I am using a right-major layout
    // start increasing the last index
    size_t index = stride.size() - 1;
    // as long as the current index has reached 0,
    // set it to width-1 and decrease the next index
    while (idx[index] == 0 && index > 0)
    {
      idx[index] = width[index] - 1;
      ptr + idx[index] * stride[index];
      --index;
    }
    --idx[index];
    ptr -= stride[index];
    return *this;
  }
  iterator operator--(int)
  {
    iterator result(*this);
    operator--();
    return result;
  }

  iterator &operator-=(int v)
  {
    if (v < 0)
      return operator+=(-v);
    size_t index = stride.size() - 1;
    while (v > 0 && index >= 0)
    {
      size_t val = v % width[index];
      v /= width[index];
      if (val <= idx[index])
      {
        idx[index] -= val;
        ptr -= val * stride[index];
      }
      else
      {
        --v;
        idx[index] += width[index] - val;
        ptr += (width[index] - val) * stride[index];
      }
      --index;
    }
    return *this;
  }

  iterator &operator+=(int v)
  {
    if (v < 0)
      return operator-=(-v);
    size_t index = stride.size() - 1;
    while (v > 0 && index >= 0)
    {
      size_t val = v % width[index];
      v /= width[index];
      idx[index] += val;
      ptr += val * stride[index];
      if (idx[index] >= width[index] && index > 0)
      {
        idx[index] -= width[index];
        ++idx[index - 1];
        ptr += stride[index - 1] - width[index] * stride[index];
      }
      --index;
    }
    return *this;
  }

  iterator operator+(int v) const
  {
    iterator result(*this);
    result += v;
    return result;
  }
  iterator operator-(int v) const
  {
    iterator result(*this);
    result -= v;
    return result;
  }

  T &operator[](int v) const
  {
    iterator iter(*this);
    iter += v;
    return *iter;
  }

  // defines equality as external friend function
  // inequality gest automatically defined by std::rel_ops
  friend bool operator==(const iterator &i, const iterator &j)
  {
    return i.ptr == j.ptr;
  }

  friend class tensor<T, type>;

private:
  iterator(const typename type::width_type &w,
           const typename type::index_type &s, T *p)
      : width(w), stride(s), idx(s), ptr(p)
  {
    std::fill(idx.begin(), idx.end(), 0);
  }

  // maintain references to width and strides
  // uses policy for acual types
  const typename type::width_type &width;
  const typename type::index_type &stride;

  // maintains both indices and pointer to data
  // uses pointer to data for dereference and equality for efficiency
  typename type::index_type idx;
  T *ptr;
};

// iterator over single index
// does not need to know actual tensor type
template <typename T>
class index_iterator
{
public:
  T &operator*() const { return *ptr; }

  index_iterator &operator++()
  {
    ptr += stride;
    return *this;
  }
  index_iterator operator++(int)
  {
    index_iterator result(*this);
    operator++();
    return result;
  }
  index_iterator &operator--()
  {
    ptr -= stride;
    return *this;
  }
  index_iterator operator--(int)
  {
    index_iterator result(*this);
    operator--();
    return result;
  }

  index_iterator &operator-=(int v)
  {
    ptr -= v * stride;
    return *this;
  }
  index_iterator &operator+=(int v)
  {
    ptr + -v *stride;
    return *this;
  }

  index_iterator operator+(int v) const
  {
    index_iterator result(*this);
    result += v;
    return result;
  }
  index_iterator operator-(int v) const
  {
    index_iterator result(*this);
    result -= v;
    return result;
  }

  T &operator[](int v) const { return *(ptr + v * stride); }

  friend bool operator==(const index_iterator &i, const index_iterator &j)
  {
    return i.ptr == j.ptr;
  }

  template <typename, typename>
  friend class ::tensor::tensor;

private:
  index_iterator(size_t s, T *p) : stride(s), ptr(p) {}

  size_t stride;
  T *ptr;
};

// Build occurrences vector for a given vector of char
std::map<char, int> occurrences(const std::vector<char> &chars)
{
  std::map<char, int> occurrences;
  for (auto &&c : chars)
  {
    auto found = occurrences.find(c);
    if (found == occurrences.end())
    {
      occurrences[c] = 0;
    }

    occurrences[c]++;
  }

  return occurrences;
}

// Get single-occurrences in an array of char
std::vector<char> calc_free_indices(const std::vector<char> &indices)
{
  auto occ = occurrences(indices);

  std::vector<char> indicesNew;
  for (auto &&i : occ)
  {
    if (i.second == 1)
    {
      indicesNew.push_back(i.first);
    }
  }

  return indicesNew;
}

// Get multiple-occurrences in a vector of char
std::vector<char> calc_dummy_indices(const std::vector<char> &indices)
{
  auto occ = occurrences(indices);

  std::vector<char> res;
  for (auto &&i : occ)
  {
    if (i.second > 1)
    {
      res.push_back(i.first);
    }
  }
  return res;
}

// Checks that the dimensions of the indexes are the same for the same index character
// Cannot be done statically
void find_incongruences(const std::vector<char> &indices, const std::vector<size_t> &width)
{
  std::vector<char> dummy_idx(calc_dummy_indices(indices));
  //    assert(indices.size() == width.size());
  for (auto idx : dummy_idx)
  {
    int dim = 0;
    size_t i = 0;
    while (dim == 0 && i < width.size())
    {
      if (indices[i] == idx)
      {
        dim = width[i];
      }
      i++;
    }

    for (i = 0; i < width.size(); ++i)
    {
      if (indices[i] == idx)
      {
        assert(dim == width[i]);
      }
    }
  }
}

// Remove duplicates in an array
std::vector<char> setify(const std::vector<char> &indices)
{
  auto occ = occurrences(indices);

  std::vector<char> indicesNew;
  for (auto &&i : occ)
  {
    indicesNew.push_back(i.first);
  }

  return indicesNew;
}

// Calculate strides given widths
std::vector<size_t> calc_strides(const std::vector<size_t> &dimensions)
{
  std::vector<size_t> strides;
  size_t stride = 1;
  size_t i = 0;

  for (auto &&dim : dimensions)
  {
    strides.push_back(stride);
    stride *= dim;
    ++i;
  }

  return strides;
}

// Calculate multi-dimension index given a linear index and widths
std::vector<size_t> build_index(size_t i, const std::vector<size_t> &width)
{
  std::vector<size_t> stride = calc_strides(width);
  std::vector<size_t> result;

  for (size_t k = 0; k < stride.size(); k++)
  {
    result.push_back((i / stride[k]) % width[k]);
  }

  return result;
}

// Construct a new multi-dimension index from index letters
std::vector<size_t> swap_indices(const std::vector<char> &source_chars,
                                 const std::vector<size_t> &source_indices,
                                 const std::vector<char> &dest_chars)
{
  std::vector<size_t> dest;

  for (size_t i = 0; i < source_chars.size(); i++)
  {
    for (size_t j = 0; j < dest_chars.size(); j++)
    {
      if (source_chars.at(i) == dest_chars.at(j))
      {
        dest.push_back(source_indices.at(i));
      }
    }
  }

  return dest;
}

template <typename T, typename FT, typename ST, typename OP>
class tensor_op;

template <typename T>
struct op_sum
{
  static T apply(T a, T b) { return a + b; }
};
template <typename T>
struct op_mult
{
  static T apply(T a, T b) { return a * b; }
};

template <typename T>
class tensor_constant
{
public:
  const tensor<T> &tensorRef;
  std::vector<char> indices;

  tensor_constant(const tensor<T> &tensorRef, const std::vector<char> &indices)
      : tensorRef(tensorRef), indices(indices)
  {
    // fails assert if there are more or less indexes than dimension of the tensor
    assert(indices.size() == tensorRef.get_rank());
    find_incongruences(indices, tensorRef.width);
  }

  tensor<T> evaluate()
  {
    if (free_indices().size() == indices.size())
    {
      return tensorRef;
    }
    else
    {
      return evaluate_repeated();
    }
  }

  tensor<T> evaluate_repeated()
  {
    auto all_indices = setify(indices);
    auto all_dims = calc_dimensions(all_indices);
    size_t total_count = std::accumulate(all_dims.begin(), all_dims.end(), 1,
                                         std::multiplies<double>());

    auto dest_indices = free_indices();
    if (dest_indices.size() > 0)
    {
      auto dest_dims = calc_dimensions(dest_indices);
      tensor<T> new_tensor(dest_dims);

      for (size_t i = 0; i < total_count; i++)
      {
        std::vector<size_t> dim_index = build_index(i, all_dims);
        std::vector<size_t> dest_index =
            swap_indices(indices, dim_index, dest_indices);

        new_tensor(dest_index) += tensorRef(dim_index);
      }

      return new_tensor;
    }
    else
    {
      // if the dimensions are =0, return a scalar instead
      // (encapsulated in a 1-rank 1-element tensor)
      T dest_scalar = 0;

      for (size_t i = 0; i < total_count; i++)
      {
        std::vector<size_t> dim_index = build_index(i, all_dims);
        std::vector<size_t> tensor_index =
            swap_indices(all_indices, dim_index, indices);

        dest_scalar += tensorRef(tensor_index);
      }

      tensor<T> new_tensor(1);
      new_tensor(0) = dest_scalar;
      return new_tensor;
    }
  }

  std::vector<size_t> calc_dimensions(const std::vector<char> &indices)
  {
    std::vector<size_t> dims;

    for (size_t i = 0; i < indices.size(); i++)
    {
      auto dim_fst = index_dimension(indices.at(i));

      if (dim_fst > 0)
      {
        dims.push_back(dim_fst);
      }
      else
      {
        // unexpected error!
        dims.push_back(0);
      }
    }

    return dims;
  }

  std::vector<char> free_indices() { return calc_free_indices(indices); }

  size_t index_dimension(char index)
  {
    size_t i = 0;
    while (i < indices.size() && indices.at(i) != index)
    {
      i++;
    }

    if (i < indices.size())
    {
      return tensorRef.width[i];
    }
    else
    {
      // unexpected error!
      return 0;
    }
  }

  template <typename ST>
  tensor_op<T, tensor_constant<T>, ST, op_sum<T>> operator+(const ST &other)
  {
    return tensor_op<T, tensor_constant<T>, ST, op_sum<T>>(*this, other);
  }

  template <typename ST>
  tensor_op<T, tensor_constant<T>, ST, op_mult<T>> operator*(const ST &other)
  {
    return tensor_op<T, tensor_constant<T>, ST, op_mult<T>>(*this, other);
  }
};

template <typename T, typename FT, typename ST, typename OP>
class tensor_op
{
public:
  FT fst;
  ST snd;

  tensor_op(const FT &fst, const ST &snd) : fst(fst), snd(snd) {}

  tensor<T> evaluate()
  {
    auto fst_val = fst.evaluate();
    auto snd_val = snd.evaluate();
    auto fst_indices = fst.free_indices();
    auto snd_indices = snd.free_indices();
    auto indices = all_indices();
    auto all_dims = calc_dimensions(indices);
    size_t total_count = std::accumulate(all_dims.begin(), all_dims.end(), 1,
                                         std::multiplies<double>());

    auto dest_indices = free_indices();

    if (dest_indices.size() > 0)
    {
      auto dest_dims = calc_dimensions(dest_indices);

      // std::cout << "INDICES: ";
      // for (auto&& i : dest_indices) {
      //   std::cout << i << ", ";
      // }
      // std::cout << std::endl;

      // std::cout << "DIMS: ";
      // for (auto&& i : dest_dims) {
      //   std::cout << i << ", ";
      // }
      // std::cout << std::endl;

      tensor<T> new_tensor(dest_dims);

      for (size_t i = 0; i < total_count; i++)
      {
        std::vector<size_t> dim_index = build_index(i, all_dims);
        std::vector<size_t> dest_index =
            swap_indices(indices, dim_index, dest_indices);
        std::vector<size_t> fst_index =
            swap_indices(indices, dim_index, fst_indices);
        std::vector<size_t> snd_index =
            swap_indices(indices, dim_index, snd_indices);

        new_tensor(dest_index) +=
            OP::apply(fst_val(fst_index), snd_val(snd_index));
      }

      return new_tensor;
    }
    else
    {
      // if the dimensions are =0, return a scalar instead
      // (encapsulated in a 1-rank 1-element tensor)
      T dest_scalar = 0;

      for (size_t i = 0; i < total_count; i++)
      {
        std::vector<size_t> dim_index = build_index(i, all_dims);
        std::vector<size_t> fst_index =
            swap_indices(indices, dim_index, fst_indices);
        std::vector<size_t> snd_index =
            swap_indices(indices, dim_index, snd_indices);

        dest_scalar += OP::apply(fst_val(fst_index), snd_val(snd_index));
      }

      tensor<T> new_tensor(1);
      new_tensor(0) = dest_scalar;
      return new_tensor;
    }
  }

  std::vector<size_t> calc_dimensions(const std::vector<char> &indices)
  {
    std::vector<size_t> dims;

    for (size_t i = 0; i < indices.size(); i++)
    {
      auto dim_fst = fst.index_dimension(indices.at(i));
      auto dim_snd = snd.index_dimension(indices.at(i));

      if (dim_fst > 0)
      {
        dims.push_back(dim_fst);
      }
      else if (dim_snd > 0)
      {
        dims.push_back(dim_snd);
      }
      else
      {
        // unexpected error!
        dims.push_back(0);
      }
    }

    return dims;
  }

  std::vector<char> joined_indices()
  {
    auto indicesFst = fst.free_indices();
    auto indicesSnd = snd.free_indices();

    std::vector<char> joined;
    joined.reserve(indicesFst.size() + indicesSnd.size());
    joined.insert(joined.end(), indicesFst.begin(), indicesFst.end());
    joined.insert(joined.end(), indicesSnd.begin(), indicesSnd.end());

    return joined;
  }

  std::vector<char> free_indices()
  {
    return calc_free_indices(joined_indices());
  }

  std::vector<char> all_indices() { return setify(joined_indices()); }

  size_t index_dimension(char index)
  {
    auto dest_indices = free_indices();
    if (dest_indices.size() > 0)
    {
      auto dest_dims = calc_dimensions(dest_indices);

      size_t i = 0;
      while (i < dest_indices.size() && dest_indices.at(i) != index)
      {
        i++;
      }

      if (i < dest_indices.size())
      {
        return dest_dims[i];
      }
      else
      {
        // unexpected error!
        return 0;
      }
    }
    else
    {
      return 1;
    }
  }

  template <typename ST2>
  tensor_op<T, tensor_op<T, FT, ST, OP>, ST2, op_sum<T>> operator+(
      const ST2 &other)
  {
    return tensor_op<T, tensor_op<T, FT, ST, OP>, ST2, op_sum<T>>(*this, other);
  }

  template <typename ST2>
  tensor_op<T, tensor_op<T, FT, ST, OP>, ST2, op_mult<T>> operator*(
      const ST2 &other)
  {
    return tensor_op<T, tensor_op<T, FT, ST, OP>, ST2, op_mult<T>>(*this,
                                                                   other);
  }
};

} // namespace reserved

// tensor specialization for dynamic rank
template <typename T>
class tensor<T, dynamic>
{
public:
  reserved::tensor_constant<T> ein(const std::string &indices)
  {
    return reserved::tensor_constant<T>(
        *this, std::vector<char>(indices.begin(), indices.end()));
  }

  // C-style constructor with explicit rank and pointer to array of dimensions
  // all other constructors are redirected to this one
  tensor(size_t rank, const size_t dimensions[])
      : width(dimensions, dimensions + rank), stride(rank, 1UL)
  {
    for (size_t i = width.size() - 1UL; i != 0; --i)
      stride[i - 1] = stride[i] * width[i];
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }
  tensor(const std::vector<size_t> &dimensions)
      : tensor(dimensions.size(), &dimensions[0]) {}
  tensor(std::initializer_list<size_t> dimensions)
      : tensor(dimensions.size(), &*dimensions.begin()) {}

  template <size_t rank>
  tensor(const size_t dims[rank]) : tensor(rank, dims) {}
  template <typename... Dims>
  tensor(Dims... dims)
      : width({static_cast<const size_t>(dims)...}),
        stride(sizeof...(dims), 1UL)
  {
    for (size_t i = width.size() - 1UL; i != 0UL; --i)
      stride[i - 1] = stride[i] * width[i];
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }

  tensor(const tensor<T, dynamic> &X) = default;
  tensor(tensor<T, dynamic> &&X) = default;
  tensor<T, dynamic> &operator=(const tensor<T, dynamic> &X) = default;
  tensor<T, dynamic> &operator=(tensor<T, dynamic> &&X) = default;
  // tensor<T, dynamic>& operator=(tensor_constant<T> exp){};
  // all tensor types are friend
  // this are used by alien copy constructors, i.e. copy constructors copying
  // different tensor types.
  template <typename, typename>
  friend class tensor;

  template <size_t R>
  tensor(const tensor<T, rank<R>> &X)
      : data(X.data),
        width(X.width.begin(), X.width.end()),
        stride(X.stride.begin(), X.stride.end()),
        start_ptr(X.start_ptr) {}

  // rank accessor
  size_t get_rank() const { return width.size(); }

  // direct accessors. Similarly to std::vector, operator () does not perform
  // range check while at() does
  T &operator()(const size_t dimensions[]) const
  {
    const size_t rank = width.size();
    T *ptr = start_ptr;
    for (size_t i = 0; i != rank; ++i)
      ptr += dimensions[i] * stride[i];
    return *ptr;
  }
  T &at(const size_t dimensions[]) const
  {
    const size_t rank = width.size();
    T *ptr = start_ptr;
    for (size_t i = 0; i != rank; ++i)
    {
      assert(dimensions[i] < width[i]);
      ptr += dimensions[i] * stride[i];
    }
    return *ptr;
  }

  T &operator()(const std::vector<size_t> &dimensions) const
  {
    assert(dimensions.size() == get_rank());
    return operator()(&dimensions[0]);
  }
  T &at(const std::vector<size_t> &dimensions) const
  {
    assert(dimensions.size() == get_rank());
    return at(&dimensions[0]);
  }

  template <size_t rank>
  T &operator()(const size_t dimensions[rank]) const
  {
    assert(rank == get_rank());
    return operator()(static_cast<const size_t *>(dimensions));
  }
  template <size_t rank>
  T &at(const size_t dimensions[rank]) const
  {
    assert(rank == get_rank());
    return at(static_cast<const size_t *>(dimensions));
  }

  template <typename... Dims>
  T &operator()(Dims... dimensions) const
  {
    assert(sizeof...(dimensions) == get_rank());
    return operator()({static_cast<const size_t>(dimensions)...});
  }
  template <typename... Dims>
  T &at(Dims... dimensions) const
  {
    assert(sizeof...(dimensions) == get_rank());
    return at({static_cast<const size_t>(dimensions)...});
  }

  // slice operation create a new tensor type sharing the data and removing the
  // sliced index
  tensor<T, dynamic> slice(size_t index, size_t i) const
  {
    const size_t rank = width.size();
    assert(index < rank);
    tensor<T, dynamic> result;
    result.data = data;
    result.width.insert(result.width.end(), width.begin(),
                        width.begin() + index);
    result.width.insert(result.width.end(), width.begin() + index + 1,
                        width.end());
    result.stride.insert(result.stride.end(), stride.begin(),
                         stride.begin() + index);
    result.stride.insert(result.stride.end(), stride.begin() + index + 1,
                         stride.end());
    result.start_ptr = start_ptr + i * stride[index];

    return result;
  }
  // operator [] slices the first (leftmost) index
  tensor<T, dynamic> operator[](size_t i) const { return slice(0, i); }

  // window operation on a single index
  tensor<T, dynamic> window(size_t index, size_t begin, size_t end) const
  {
    tensor<T, dynamic> result(*this);
    result.width[index] = end - begin;
    result.start_ptr += result.stride[index] * begin;
    return result;
  }

  // window operations on all indices
  tensor<T, dynamic> window(const size_t begin[], const size_t end[]) const
  {
    tensor<T, dynamic> result(*this);
    const size_t r = get_rank();
    for (size_t i = 0; i != r; ++i)
    {
      result.width[i] = end[i] - begin[i];
      result.start_ptr += result.stride[i] * begin[i];
    }
    return result;
  }
  tensor<T, dynamic> window(const std::vector<size_t> &begin,
                            const std::vector<size_t> &end) const
  {
    return window(&(begin[0]), &(end[0]));
  }

  // flaten operation
  // do not use over windowed and sliced ranges
  tensor<T, dynamic> flatten(size_t begin, size_t end) const
  {
    tensor<T, dynamic> result;
    result.stride.insert(result.stride.end(), stride.begin(),
                         stride.begin() + begin);
    result.stride.insert(result.stride.end(), stride.begin() + end,
                         stride.end());
    result.width.insert(result.width.end(), width.begin(),
                        width.begin() + begin);
    result.width.insert(result.width.end(), width.begin() + end,
                        width.end());
    for (size_t i = begin; i != end; ++i)
      result.width[end] *= width[i];
    result.start_ptr = start_ptr;
    result.data = data;
    return result;
  }

  // specialized iterator type
  typedef reserved::iterator<T, dynamic> iterator;

  iterator begin() const { return iterator(width, stride, start_ptr); }
  iterator end() const
  {
    iterator result = begin();
    result.idx[0] = width[0];
    result.ptr += width[0] * stride[0];
    return result;
  }

  // specialized index_iterator type
  typedef reserved::index_iterator<T> index_iterator;

  // begin and end methods producing index_iterator require the index to be
  // iterated over and all the values for the other indices
  index_iterator begin(size_t index, const size_t dimensions[]) const
  {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index, const size_t dimensions[]) const
  {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  template <size_t rank>
  index_iterator begin(size_t index, const size_t dimensions[rank]) const
  {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  template <size_t rank>
  index_iterator end(size_t index, const size_t dimensions[rank]) const
  {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  index_iterator begin(size_t index,
                       const std::vector<size_t> &dimensions) const
  {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index,
                     const std::vector<size_t> &dimensions) const
  {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  //  private:
  tensor() = default;

  std::shared_ptr<std::vector<T>> data;
  dynamic::width_type width;
  dynamic::index_type stride;
  T *start_ptr;
};

// tensor specialization for fixed-rank
template <typename T, size_t R>
class tensor<T, rank<R>>
{
public:
  // C-style constructor with implicit rank and pointer to array of dimensions
  // all other constructors are redirected to this one
  tensor(const size_t dimensions[R])
  {
    size_t *wptr = &(width[0]), *endp = &(width[0]) + R;
    while (wptr != endp)
      *(wptr++) = *(dimensions++);
    stride[R - 1] = 1;
    for (int i = R - 1; i != 0; --i)
    {
      stride[i - 1] = stride[i] * width[i];
    }
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }

  tensor(const std::vector<size_t> &dimensions) : tensor(&dimensions[0])
  {
    assert(dimensions.size() == R);
  }
  template <typename... Dims>
  tensor(Dims... dims) : width({static_cast<const size_t>(dims)...})
  {
    static_assert(sizeof...(dims) == R, "size mismatch");

    stride[R - 1] = 1UL;
    for (size_t i = R - 1UL; i != 0UL; --i)
    {
      stride[i - 1] = stride[i] * width[i];
    }
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }

  tensor(const tensor<T, rank<R>> &X) = default;
  tensor(tensor<T, rank<R>> &&X) = default;

  // all tensor types are friend
  // this are used by alien copy constructors, i.e. copy constructors copying
  // different tensor types.
  template <typename, typename>
  friend class tensor;

  tensor(const tensor<T, dynamic> &X)
      : data(X.data),
        width(X.width.begin(), X.width.end()),
        stride(X.stride.begin(), X.stride.end()),
        start_ptr(X.start_ptr)
  {
    assert(X.get_rank() == R);
  }

  // not static so that it can be called with . rather than ::
  constexpr size_t get_rank() const { return R; }

  // direct accessors as for dynamic tensor
  T &operator()(const size_t dimensions[R]) const
  {
    T *ptr = start_ptr;
    for (size_t i = 0; i != R; ++i)
      ptr += dimensions[i] * stride[i];
    return *ptr;
  }
  T &at(const size_t dimensions[R]) const
  {
    T *ptr = start_ptr;
    for (size_t i = 0; i != R; ++i)
    {
      assert(dimensions[i] < width[i]);
      ptr += dimensions[i] * stride[i];
    }
    return *ptr;
  }

  T &operator()(const std::vector<size_t> &dimensions) const
  {
    assert(dimensions.size() == R);
    return operator()(&dimensions[0]);
  }
  T &at(const std::vector<size_t> &dimensions) const
  {
    assert(dimensions.size() == R);
    return at(&dimensions[0]);
  }

  // could use std::enable_if rather than static assert!
  template <typename... Dims>
  T &operator()(Dims... dimensions) const
  {
    static_assert(sizeof...(dimensions) == R, "rank mismatch");
    return operator()({static_cast<const size_t>(dimensions)...});
  }
  template <typename... Dims>
  T &at(Dims... dimensions) const
  {
    static_assert(sizeof...(dimensions) == R, "rank mismatch");
    return at({static_cast<const size_t>(dimensions)...});
  }

  // specialized iterator type
  typedef reserved::iterator<T, rank<R>> iterator;

  iterator begin() { return iterator(width, stride, start_ptr); }
  iterator end()
  {
    iterator result = begin();
    result.idx[0] = width[0];
    result.ptr += width[0] * stride[0];
    return result;
  }

  // specialized index_iterator type
  typedef reserved::index_iterator<T> index_iterator;

  index_iterator begin(size_t index, const size_t dimensions[R]) const
  {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index, const size_t dimensions[R]) const
  {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  index_iterator begin(size_t index,
                       const std::vector<size_t> &dimensions) const
  {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index,
                     const std::vector<size_t> &dimensions) const
  {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  // slicing operations return lower-rank tensors
  tensor<T, rank<R - 1>> slice(size_t index, size_t i) const
  {
    assert(index < R);
    tensor<T, rank<R - 1>> result;
    result.data = data;
    for (size_t i = 0; i != index; ++i)
    {
      result.width[i] = width[i];
      result.stride[i] = stride[i];
    }
    for (size_t i = index; i != R - 1U; ++i)
    {
      result.width[i] = width[i + 1];
      result.stride[i] = stride[i + 1];
    }
    result.start_ptr = start_ptr + i * stride[index];

    return result;
  }
  tensor<T, rank<R - 1>> operator[](size_t i) const { return slice(0, i); }

  // window operations do not change rank
  tensor<T, rank<R>> window(size_t index, size_t begin, size_t end) const
  {
    tensor<T, rank<R>> result(*this);
    result.width[index] = end - begin;
    result.start_ptr += result.stride[index] * begin;
    return result;
  }

  tensor<T, rank<R>> window(const size_t begin[], const size_t end[]) const
  {
    tensor<T, rank<R>> result(*this);
    for (size_t i = 0; i != R; ++i)
    {
      result.width[i] = end[i] - begin[i];
      result.start_ptr += result.stride[i] * begin[i];
    }
    return result;
  }
  tensor<T, dynamic> window(const std::vector<size_t> &begin,
                            const std::vector<size_t> &end) const
  {
    return window(&begin[0], &end[0]);
  }

  // flatten operations change rank in a way that is not known at compile time
  // would need a different interface to provide that info at compile time,
  // but the operation should not be time-critical
  tensor<T, dynamic> flatten(size_t begin, size_t end) const
  {
    tensor<T, dynamic> result;
    result.stride.insert(result.stride.end(), stride.begin(),
                         stride.begin() + begin);
    result.stride.insert(result.stride.end(), stride.begin() + end,
                         stride.end());
    result.width.insert(result.width.end(), width.begin(),
                        width.begin() + begin);
    result.stride.insert(result.stride.end(), stride.begin() + end,
                         stride.end());
    for (size_t i = begin; i != end; ++i)
      result.width[end] *= width[i];
    result.start_prt = start_ptr;
    result.data = data;
    return result;
  }

  friend class tensor<T, rank<R + 1>>;

private:
  tensor() = default;

  std::shared_ptr<std::vector<T>> data;
  typename rank<R>::width_type width;
  typename rank<R>::index_type stride;
  T *start_ptr;
};

// tensor specialization for rank 1
// in this case splicing provides reference to data element
template <typename T>
class tensor<T, rank<1>>
{
public:
  tensor(size_t dimension)
  {
    data = std::make_shared<std::vector<T>>(dimension);
    start_ptr = &*(data->begin());
  }

  // all tensor types are friend
  // this are used by alien copy constructors, i.e. copy constructors copying
  // different tensor types.
  template <typename, typename>
  friend class tensor;

  constexpr size_t get_rank() const { return 1; }

  // direct accessors as for dynamic tensor
  T &operator()(size_t d) const { return start_ptr[d * stride[0]]; }
  T &at(size_t d) const
  {
    assert(d < width[0]);
    return start_ptr[d * stride[0]];
  }

  T &operator()(const size_t dimensions[1]) const
  {
    return operator()(dimensions[0]);
  }
  T &at(const size_t dimensions[1]) const { return operator()(dimensions[0]); }

  T &operator()(const std::vector<size_t> &dimensions) const
  {
    assert(dimensions.size() == 1);
    return operator()(dimensions[0]);
  }
  T &at(const std::vector<size_t> &dimensions) const
  {
    assert(dimensions.size() == 1);
    return operator()(dimensions[0]);
  }

  // could use std::enable_if rather than static assert!

  T &slice(size_t index, size_t i) const
  {
    assert(index == 0);
    return *(start_ptr + i * stride[0]);
  }
  T &operator[](size_t i) { return *(start_ptr + i * stride[0]); }

  tensor<T, rank<1>> window(size_t begin, size_t end) const
  {
    tensor<T, rank<1>> result(*this);
    result.width[0] = end - begin;
    result.start_ptr += result.stride[0] * begin;
    return result;
  }

  typedef T *iterator;
  iterator begin(size_t = 0) { return start_ptr; }
  iterator end(size_t = 0) { return start_ptr + width[0] * stride[0]; }

  friend class tensor<T, rank<2>>;

private:
  tensor() = default;
  std::shared_ptr<std::vector<T>> data;
  rank<1>::width_type width;
  rank<1>::index_type stride;
  T *start_ptr;
};

}; // namespace tensor

#endif // TENSOR
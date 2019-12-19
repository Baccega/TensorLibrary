#include <algorithm>

#include <vector>

#include <map>

#include <iostream>

#include <cassert>

// Build occurrences vector for a given vector of char

std::map<char, int> occurrences(const std::vector<char>& chars) {

  std::map<char, int> occurrences;

  for (auto&& c : chars) {

//    auto found = occurrences.find(c);

//    if (found == occurrences.end()) {

//      occurrences[c] = 0;

//    }



    occurrences[c]++;

  }



  return occurrences;

}



// Get single-occurrences in a vector of char

std::vector<char> calc_free_indices(const std::vector<char>& indices) {

  auto occ = occurrences(indices);



  std::vector<char> res;

  for (auto&& i : occ) {

    if (i.second == 1) {

      res.push_back(i.first);

    }

  }



  return res;

}



// Get multiple-occurrences in a vector of char 

std::vector<char> calc_dummy_indices(const std::vector<char>& indices) {

  auto occ = occurrences(indices);



  std::vector<char> res;

  for (auto&& i : occ) {

    if (i.second > 1) {

      res.push_back(i.first);

    }

  }

  return res;

}



// Checks that the dimensions of the indexes are the same for the same index character

// Cannot be done statically

void find_incongruences(const std::vector<char>& indices, const std::vector<size_t>& width){

    std::vector<char> dummy_idx(calc_dummy_indices(indices));

//    assert(indices.size() == width.size());

    for (auto idx : dummy_idx){

        int dim = 0;

        size_t i = 0;

        while(dim == 0 && i < width.size() ){

            if (indices[i] == idx) {dim = width[i];}

            i++;

        }

        

        for (i = 0;i < width.size(); ++i){

            if ( indices[i] == idx){

                assert(dim == width[i]);

            }

        }

    }

}



// Remove duplicates in a vector of char

std::vector<char> setify(const std::vector<char>& indices) {

  auto occ = occurrences(indices);



  std::vector<char> res;

  for (auto&& i : occ) {

    res.push_back(i.first);

  }



  return res;

}



// Calculate strides given widths

std::vector<size_t> calc_strides(const std::vector<size_t>& dimensions) {

  std::vector<size_t> res;

  size_t stride = 1;

  size_t i = 0;



  for (auto&& dim : dimensions) {

    res.push_back(stride);

    stride *= dim;

    ++i;

  }



  return res;

}



// Calculate multi-dimension index given a linear index and widths

std::vector<size_t> build_index(size_t i, const std::vector<size_t>& width) {

  std::vector<size_t> stride = calc_strides(width);

  std::vector<size_t> result;



  for (size_t k = 0; k < stride.size(); k++) {

    result.push_back((i / stride[k]) % width[k]);

  }



  return result;

}



// Construct a new multi-dimension index from index letters

std::vector<size_t> swap_indices(const std::vector<char>& source_chars,

                                 const std::vector<size_t>& source_indices,

                                 const std::vector<char>& dest_chars) {

  std::vector<size_t> dest;



  for (size_t i = 0; i < source_chars.size(); i++) {

    for (size_t j = 0; j < dest_chars.size(); j++) {

      if (source_chars.at(i) == dest_chars.at(j)) {

        dest.push_back( source_indices.at(i) );

      }

    }

  }



  return dest;

}



int main(){

    std::vector<char> idx = {'i','j','j','k','k','l','k'};

    auto idxmap = occurrences(idx);

    std::cout<<std::endl;

    for (auto x : idxmap){

        std::cout<<'('<<x.first<<','<<x.second<<')';

    }

    std::cout<<std::endl;

    std::cout<<std::endl;

    

    auto fidx = calc_free_indices(idx);

    for (auto x : fidx){

        std::cout<<' '<<x<<' ';

    }

    std::cout<<std::endl;

    std::cout<<std::endl;



    auto all_idx = setify(idx);

    for (auto x : all_idx){

        std::cout<<' '<<x<<' ';

    }

    std::cout<<std::endl;

    std::cout<<std::endl;



    std::vector<size_t> dims = {2,3,3,4,4,5,6};

    auto strd = calc_strides(dims);

    for (auto x : strd){

        std::cout<<' '<<x<<' ';

    }

    std::cout<<std::endl;

    std::cout<<std::endl;



    auto bidx = build_index(1353,dims);

    for (auto x : bidx){

        std::cout<<' '<<x<<' ';

    }

    std::cout<<std::endl;

    std::cout<<std::endl;

    

    find_incongruences(idx, dims);

}
#ifndef ACTSEEKLIB_H
#define ACTSEEKLIB_H

#include <iostream>
#include <string>

#include <array>
#include <unordered_set>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <functional> // for std::hash
#include <memory> // for std::unique_ptr
#include <limits>

#include <algorithm> // for std::sample, std::min
#include <cmath> // for std::fabs
#include <cstdlib> // for std::rand
#include <ctime>   // for std::time
#include <iterator> // for std::back_inserter
#include <numeric> // for std::accumulate
#include <random> // for std::mt19937, std::random_device

#include <Eigen/Dense>
#include <nanoflann.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

using namespace std;
namespace py = pybind11;

extern bool* cuda_check_distances(float* coord_vecs1, float* coord_vecs2, float* distances_vecs, int* distances_offsets, int* distances_sizes, int n_vecs, int n_dim, float threshold);
extern bool is_gpu_available();
void set_gpu_enabled(bool enabled);
bool gpu_enabled();

using profile_t = tuple<string, vector<Eigen::Vector3d>, vector<Eigen::Vector3d>, unordered_map<int, string>, unordered_map<int, int>>;
// using entry_t = tuple<vector<int>, string, string, vector<Eigen::Vector3d>, vector<Eigen::Vector3d>, vector<Eigen::Vector3d>,
//     unordered_map<int, string>, vector<int>, unordered_map<int, int>>;
using result_t = tuple<string, int, int, string, string, double, double, double, double, double, string>;

// Hash function for std::tuple
// struct tuple_hash {
//     template <typename... Args>
//     std::size_t operator()(const std::tuple<Args...>& t) const {
//         return std::apply([](const auto&... args) {
//             return (... ^ (std::hash<std::decay_t<decltype(args)>>{}(args))); // Combine hashes
//         }, t);
//     }
// };
struct tuple_hash {
    template <typename... Args>
    std::size_t operator()(const std::tuple<Args...>& t) const {
        std::size_t seed = 0;
        std::apply([&seed](const auto&... args) {
            ((seed ^= std::hash<std::decay_t<decltype(args)>>{}(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2)), ...);
        }, t);
        return seed;
    }
};

using PointCloud = vector<Eigen::Vector3d>;
struct PointCloudAdapter {
    const PointCloud& cloud;

    inline size_t kdtree_get_point_count() const { return cloud.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return cloud[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdapter>,
    PointCloudAdapter,
    3>;

class Amino_acid {
public:
    Amino_acid(
        const int index, 
        const int real_index, 
        const string& aa, 
        const string& aa_class, 
        const Eigen::Vector3d& coords);

    Eigen::Vector3d get_coords() const;
    string get_class() const;
    int get_by_index() const;
    vector<double> get_distance(const string& aa_class) const;
    void set_distances(const Eigen::Vector3d& coord, const string& aa_class);

private:
    int index;
    int real_index;
    string aa;
    string aa_class;
    Eigen::Vector3d coords;
    unordered_map<string, vector<double>> distances;
};

class Active_site {
public:
    Active_site(
        const vector<int>& active,
        const unordered_map<int, string>& aa_cav,
        const vector<Eigen::Vector3d>& coords,
        const unordered_map<string, string>& aa_des);

    int get_Number_AA() const;
    vector<Amino_acid> get_AA() const;
    vector<string> get_AA_groups(bool unique) const;
    void set_distances();
    vector<double> get_distances(const string& aa_class1, const string& aa_class2);

    vector<pair<int, int>> get_possible_correspondences(
        const vector<Eigen::Vector3d>& protein,
        const unordered_map<int, string>& aa,
        const unordered_map<int, int>& real_index);

    vector<array<pair<int, int>, 3>> get_all_possible_combinations(
        const vector<pair<int, int>>& pc,
        const vector<Eigen::Vector3d>& protein_coords,
        const unordered_map<int, string>& aa,
        const unordered_map<int, int>& real_index,
        const double threshold,
        const unordered_map<string, string>& aa_des);

private:
    vector<int> active;
    unordered_map<int, string> aa_cav;
    vector<Amino_acid> aa;
    vector<Eigen::Vector3d> coords;
    unordered_map<string, string> aa_des;
    unordered_map<tuple<string, string>, vector<double>, tuple_hash> distances_dict;
    void add_amino_acids();
    bool check_dist(
        const double dist,
        const vector<double>& distances,
        const double threshold);
    double calculate_distance(
        const int index1,
        const int index2,
        const vector<Eigen::Vector3d>& protein_coords);
    bool* check_distances(
        const pair<int, int>& part,
        const vector<array<pair<int, int>, 3>>& combinations,
        const vector<Eigen::Vector3d>& protein_coords,
        const unordered_map<int, string>& aa,
        const unordered_map<int, int>& real_index,
        const double threshold,
        const unordered_map<string, string>& aa_des);
 unordered_map<tuple<int, int>, bool, tuple_hash> check_distances_on_gpu(
        const vector<pair<int, int>>& pc,
        const vector<Eigen::Vector3d>& protein_coords,
        const unordered_map<int, string>& aa,
        const unordered_map<int, int>& real_index,
        const double threshold,
        const unordered_map<string, string>& aa_des);
};

tuple<Eigen::Vector3d, Eigen::Matrix3d> estimate_alignment(
    const vector<Eigen::Vector3d>& points1,
    const vector<Eigen::Vector3d>& points2);

tuple<Eigen::Vector3d, Eigen::Matrix3d> euclidean_transformation_fit(
    const vector<Eigen::Vector3d>& protein,
    const vector<Eigen::Vector3d>& protein_cb,
    const vector<Eigen::Vector3d>& cavity,
    const vector<Eigen::Vector3d>& cavity_cb,
    const array<pair<int, int>, 3>& paa);

tuple<vector<double>, vector<pair<int, int>>> calculate_distances(
    const vector<Eigen::Vector3d>& transf,
    const vector<Eigen::Vector3d>& cavity,
    const double threshold);

tuple<vector<pair<int, int>>, double, vector<double>, vector<Eigen::Vector3d>, Eigen::Vector3d, Eigen::Matrix3d> calculate_final_distance(
    const vector<Eigen::Vector3d>& cavity,
    const vector<Eigen::Vector3d>& protein,
    const vector<Eigen::Vector3d>& protein_cb,
    const vector<array<pair<int, int>, 3>>& valid_combinations,
    const int i,
    const unordered_map<string, string>& aa_des,
    const unordered_map<int, string>& aa,
    const unordered_map<int, int>& real_index,
    const unordered_map<int, string>& aa_cav,
    const vector<int>& active,
    const vector<Eigen::Vector3d>& cavity_coords_used,
    const vector<Eigen::Vector3d>& cavity_coords_cb_used,
    const double threshold);

tuple<double, double> get_distance_around(
    const vector<Eigen::Vector3d>& new_protein_coords,
    const vector<Eigen::Vector3d>& seed_coords,
    const vector<pair<int, int>> solution,
    const unordered_map<int, int>& real_index_seed,
    const vector<int>& active,
    const double aa_surrounding);

tuple<double, vector<double>, double, vector<Eigen::Vector3d>, vector<pair<int, int>>, Eigen::Vector3d, Eigen::Matrix3d> ransac_protein(
    const vector<Eigen::Vector3d>& cavity,
    const vector<Eigen::Vector3d>& protein,
    const vector<Eigen::Vector3d>& protein_cb,
    const vector<Eigen::Vector3d>& seed_coords,
    const vector<array<pair<int, int>, 3>>& valid_combinations,
    const int iterations,
    const unordered_map<string, string>& aa_des,
    const unordered_map<int, string>& aa,
    const unordered_map<int, int>& real_index,
    const unordered_map<int, int>& real_index_seed,
    const unordered_map<int, string>& aa_cav,
    const vector<int>& active,
    const vector<Eigen::Vector3d>& cavity_coords_used,
    const vector<Eigen::Vector3d>& cavity_coords_cb_used,
    const double aa_around,
    const double threshold);

double score(vector<int> vec, double size);
double score_vector(vector<int> vec);

tuple<vector<Eigen::Vector3d>, vector<Eigen::Vector3d>, vector<double>, vector<int>> find_nearest_neighbors(
    const vector<Eigen::Vector3d>& vector1,
    const vector<Eigen::Vector3d>& vector2);

tuple<double, double, double> get_global_distance(
    const vector<Eigen::Vector3d>& coords1,
    const vector<Eigen::Vector3d>& coords2);

tuple<bool, string, string, double, double, double, double, double, string> proteinRansacMain(
    const vector<int>& index_used,
    const unordered_map<string, string>& aa_des,
    const int iterations,
    const string case_protein_name,
    const vector<Eigen::Vector3d>& protein_coords,
    const vector<Eigen::Vector3d>& protein_coords_cb,
    const vector<Eigen::Vector3d>& seed_coords,
    const vector<Eigen::Vector3d>& cavity_coords,
    const vector<Eigen::Vector3d>& cavity_coords_cb,
    const unordered_map<int, string>& aa,
    const unordered_map<int, string>& aa_cav,
    const vector<int>& active,
    const unordered_map<int, int>& real_index,
    const unordered_map<int, int>& real_index_seed);
    
result_t proteinRansacMain_t(
    const int t_id,
    const vector<int>& index_used,
    const unordered_map<string, string>& aa_des,
    const int iterations,
    const string& seed_protein,
    const string& case_protein,
    const string& ec,
    const vector<Eigen::Vector3d>& protein_coords,
    const vector<Eigen::Vector3d>& protein_coords_cb,
    const vector<Eigen::Vector3d>& seed_coords,
    const vector<Eigen::Vector3d>& cavity_coords,
    const vector<Eigen::Vector3d>& cavity_coords_cb,
    const unordered_map<int, string>& aa,
    const unordered_map<int, string>& aa_cav,
    const vector<int>& active,
    const unordered_map<int, int>& real_index,
    const unordered_map<int, int>& real_index_seed);

vector<result_t> concurrentMain(
    const profile_t& case_protein_profile,
    const string& segment_name,
    const string& table_name);

#endif

#include "ActSeekLib.h"
#include "thread_pool.h"
#include "shared_memory.h"


size_t THREAD_POOL_SIZE = 8;
bool GPU_ENABLED = true;

size_t get_thread_pool_size() {
    return THREAD_POOL_SIZE;
}

void set_thread_pool_size(size_t size) {
    THREAD_POOL_SIZE = std::max<size_t>(1, size);
}

void set_gpu_enabled(bool enabled) {
    GPU_ENABLED = enabled;
}

bool gpu_enabled() {
    return GPU_ENABLED;
}

const unordered_map<string, string> aa_des = {{"GLY", "GLY"}, {"ALA", "ALA"}, {"PRO", "PRO"}, {"ARG", "ARG"}, {"HIS", "HIS"}, {"LYS", "LYS"}, {"ASP", "ASP"},
                  {"GLU", "GLU"},
                  {"SER", "SER"}, {"THR", "THR"}, {"ASN", "ASN"}, {"GLN", "GLN"}, {"CYS", "CYS"}, {"VAL", "VAL"}, {"ILE", "ILE"},
                  {"LEU", "LEU"},
                  {"MET", "MET"}, {"PHE", "PHE"}, {"TYR", "ARO"}, {"TRP", "ARO"}, {"LIG", "DIF"}, {"MSE", "HFBd"}, {"HSD", "HIS"}};

const unordered_map<int, string> aa_names = {
    {1, "ALA"}, {2, "ARG"}, {3, "ASN"}, {4, "ASP"}, {5, "CYS"},
    {6, "GLN"}, {7, "GLU"}, {8, "GLY"}, {9, "HIS"}, {10, "ILE"},
    {11, "LEU"}, {12, "LYS"}, {13, "MET"}, {14, "PHE"}, {15, "PRO"},
    {16, "SER"}, {17, "THR"}, {18, "TRP"}, {19, "TYR"}, {20, "VAL"},
    {21, "SEC"}
};

Amino_acid::Amino_acid(
    const int index, 
    const int real_index, 
    const string& aa, 
    const string& aa_class, 
    const Eigen::Vector3d& coords)
    : index(index), real_index(real_index), aa(aa), aa_class(aa_class), coords(coords) {

}

Eigen::Vector3d Amino_acid::get_coords() const {
    return coords;
}

string Amino_acid::get_class() const {
    return aa_class;
}

int Amino_acid::get_by_index() const {
    return index;
}

vector<double> Amino_acid::get_distance(const string& aa_class) const {
    auto it = distances.find(aa_class);
    if (it != distances.end()) {
        return it->second;
    }
    return {};
}

void Amino_acid::set_distances(const Eigen::Vector3d& coord, const string& aa_class) {
    double dist = (coords - coord).norm();
    distances[aa_class].push_back(dist);
}

Active_site::Active_site(
    const vector<int>& active, 
    const unordered_map<int, string>& aa_cav, 
    const vector<Eigen::Vector3d>& coords, 
    const unordered_map<string, string>& aa_des)
    : active(active), aa_cav(aa_cav), coords(coords), aa_des(aa_des) {
    add_amino_acids();
    set_distances();
}

void Active_site::add_amino_acids() {
    for (const auto& kv : aa_cav) {
        auto it = std::find(active.begin(), active.end(), kv.first);
        if (it != active.end()) {
            int index1 = std::distance(active.begin(), it);
            auto it2 = aa_des.find(kv.second);
            if (it2 != aa_des.end()) {
                Amino_acid aminoacid(index1, kv.first, kv.second, aa_des.at(kv.second), coords[index1]);
                aa.push_back(aminoacid);
            }
            else {
                //cerr << kv.second << " is not found in aa_des!" << endl;
            }
        }
        else {
            //cerr << kv.first << " is not found in active!" << endl;
        }
    }
}

int Active_site::get_Number_AA() const {
    return aa.size();
}

vector<Amino_acid> Active_site::get_AA() const {
    return aa;
}

vector<string> Active_site::get_AA_groups(bool unique) const {
    vector<string> groups;
    for (const auto& aa1 : aa) {
        groups.push_back(aa1.get_class());
    }
    if (unique) {
        sort(groups.begin(), groups.end());
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    }
    return groups;
}

void Active_site::set_distances() {
    for (auto& aa1 : aa) {
        for (auto& aa2 : aa) {
            if (&aa1 != &aa2) {
                aa1.set_distances(aa2.get_coords(), aa2.get_class());
            }
        }
    }
}

vector<double> Active_site::get_distances(const string& aa_class1, const string& aa_class2) {
    tuple<string, string> key = make_tuple(aa_class1, aa_class2);
    auto it = distances_dict.find(key);
    if (it != distances_dict.end()) {
        return it->second;
    } else {
        vector<double> dist;
        for (const auto& aa1 : aa) {
            if (aa1.get_class() == aa_class1) {
                vector<double> distances = aa1.get_distance(aa_class2);
                dist.insert(dist.end(), distances.begin(), distances.end());
            }
        }
        std::sort(dist.begin(), dist.end());
        distances_dict.try_emplace(key, dist);
        return dist;
    }
}

vector<pair<int, int>> Active_site::get_possible_correspondences(
    const vector<Eigen::Vector3d>& protein, 
    const unordered_map<int, string>& aa_protein, 
    const unordered_map<int, int>& real_index) {
    vector<pair<int, int>> correspondencies;
    for (int aa1 = 0; aa1 < protein.size(); ++aa1) {
        for (const auto& aa2 : aa) {
            try {
                if (aa_des.at(aa_protein.at(real_index.at(aa1))) == aa2.get_class()) {
                    correspondencies.push_back({ aa1, aa2.get_by_index() });
                }
            } catch(...) {
                cerr << "No aa" << endl;
            }
        }
    }
    return correspondencies;
}

bool Active_site::check_dist(
    const double dist, 
    const vector<double>& distances, 
    const double threshold) {
    // for (double d : distances) {
    //     if (std::fabs(dist - d) <= threshold) {
    //         return true;
    //     }
    // }
    auto it = std::lower_bound(distances.begin(), distances.end(), dist - threshold);
    if (it != distances.end() && *it <= dist + threshold) {
        return true;
    }
    return false;
}

double Active_site::calculate_distance(
    const int index1, 
    const int index2, 
    const vector<Eigen::Vector3d>& protein_coords) {
    return (protein_coords[index1] - protein_coords[index2]).norm();
}

bool* Active_site::check_distances(
    const pair<int, int>& part,
    const vector<array<pair<int, int>, 3>>& combinations,
    const vector<Eigen::Vector3d>& protein_coords,
    const unordered_map<int, string>& aa,
    const unordered_map<int, int>& real_index,
    const double threshold,
    const unordered_map<string, string>& aa_des) {
    bool* results = new bool[combinations.size()];
    for (int i = 0; i < combinations.size(); ++i) {
        double dist = calculate_distance(combinations[i][part.first].first, combinations[i][part.second].first, protein_coords);
        vector<double> distances = get_distances(aa_des.at(aa.at(real_index.at(combinations[i][part.first].first))), aa_des.at(aa.at(real_index.at(combinations[i][part.second].first))));
        if (check_dist(dist, distances, threshold)) {
            results[i] = true;
        } else {
            results[i] = false;
        }
    }
    return results;
}

unordered_map<tuple<int, int>, bool, tuple_hash> Active_site::check_distances_on_gpu(
    const vector<pair<int, int>>& pc,
    const vector<Eigen::Vector3d>& protein_coords,
    const unordered_map<int, string>& aa,
    const unordered_map<int, int>& real_index,
    const double threshold,
    const unordered_map<string, string>& aa_des) {
    auto flush_dist_cache = [&](
        const int a,
        const int b,
        unordered_map<tuple<int, int>, bool, tuple_hash>& dist_cache) {
        tuple<int, int> key = make_tuple(a, b);
        auto it = dist_cache.find(key);
        if (it == dist_cache.end()) {
            dist_cache.try_emplace(key, true);
        }
    };
    auto check_dist_cache = [&](
        const int a,
        const int b,
        const unordered_map<tuple<int, int>, bool, tuple_hash>& dist_cache) {
        tuple<int, int> key = make_tuple(a, b);
        auto it = dist_cache.find(key);
        if (it != dist_cache.end()) {
            return it->second;
        } else {
            return false;
        }
    };
    auto check_dist_cache_on_gpu = [&](unordered_map<tuple<int, int>, bool, tuple_hash>& dist_cache) {
        int n_vecs = static_cast<int>(dist_cache.size());
        int n_dim = 3;
        vector<float> coord_vecs1(n_vecs * n_dim), coord_vecs2(n_vecs * n_dim);
        vector<float> distances_vecs;
        vector<int> distances_offsets;
        vector<int> distances_sizes;
        int distances_offset = 0;
        int vec_i = 0;
        for (const auto& [k, v] : dist_cache) {
            vector<double> distances = get_distances(aa_des.at(aa.at(real_index.at(get<0>(k)))), aa_des.at(aa.at(real_index.at(get<1>(k)))));
            std::transform(distances.begin(), distances.end(), std::back_inserter(distances_vecs), [](double val) { return static_cast<float>(val); });
            distances_offsets.push_back(distances_offset);
            distances_sizes.push_back(static_cast<int>(distances.size()));
            distances_offset += static_cast<int>(distances.size());

            for (int coord_i = 0; coord_i < n_dim; ++coord_i) {
                coord_vecs1[vec_i * n_dim + coord_i] = static_cast<float>(protein_coords[get<0>(k)][coord_i]);
                coord_vecs2[vec_i * n_dim + coord_i] = static_cast<float>(protein_coords[get<1>(k)][coord_i]);
            }
            ++vec_i;
        }
        bool* results = cuda_check_distances(coord_vecs1.data(), coord_vecs2.data(), distances_vecs.data(), distances_offsets.data(), distances_sizes.data(), n_vecs, n_dim, static_cast<float>(threshold));
        vec_i = 0;
        for (auto& [k, v] : dist_cache) {
            v = results[vec_i];
            ++vec_i;
        }
        delete[] results;
    };
    unordered_map<tuple<int, int>, bool, tuple_hash> dist_cache1;
    for (int i = 0; i < pc.size(); ++i) {
        for (int j = i + 1; j < pc.size(); ++j) {
            if (pc[i].second == pc[j].second) {
                continue;
            }
            flush_dist_cache(pc[i].first, pc[j].first, dist_cache1);
        }
    }
    check_dist_cache_on_gpu(dist_cache1);
    unordered_map<tuple<int, int>, bool, tuple_hash> dist_cache2;
    for (int i = 0; i < pc.size(); ++i) {
        for (int j = i + 1; j < pc.size(); ++j) {
            if (pc[i].second == pc[j].second) {
                continue;
            }
            check_dist_cache(pc[i].first, pc[j].first, dist_cache1);
            for (int k = j + 1; k < pc.size(); ++k) {
                // set<int> distinct_classes = { pc[i].second, pc[j].second, pc[k].second };
                // if (distinct_classes.size() < 3) {
                //     continue;
                // }
                if ((pc[j].second == pc[k].second) || (pc[k].second == pc[i].second)) {
                    continue;
                }
                flush_dist_cache(pc[j].first, pc[k].first, dist_cache2);
                flush_dist_cache(pc[k].first, pc[i].first, dist_cache2);
            }
        }
    }
    check_dist_cache_on_gpu(dist_cache2);
    for (const auto& [k, v] : dist_cache2) {
        dist_cache1[k] = v;
    }
    return dist_cache1;
}

vector<array<pair<int, int>, 3>> Active_site::get_all_possible_combinations(
    const vector<pair<int, int>>& pc,
    const vector<Eigen::Vector3d>& protein_coords,
    const unordered_map<int, string>& aa,
    const unordered_map<int, int>& real_index,
    const double threshold,
    const unordered_map<string, string>& aa_des) {
    unordered_map<tuple<int, int>, bool, tuple_hash> dist_cache;
    vector<array<pair<int, int>, 3>> valid_combinations;
    if (gpu_enabled() && is_gpu_available() && pc.size() > 1000) {
        dist_cache = check_distances_on_gpu(pc, protein_coords, aa, real_index, threshold, aa_des);
    } 
    // else {
    auto check_dist_cache = [&](const int a, const int b, const string& aa_des_a, const string& aa_des_b) {
        tuple<int, int> key = make_tuple(a, b);
        auto it = dist_cache.find(key);
        if (it != dist_cache.end()) {
            return it->second;
        } else {
            double dist = calculate_distance(a, b, protein_coords);
            vector<double> distances = get_distances(aa_des_a, aa_des_b);
            bool result = check_dist(dist, distances, threshold);
            dist_cache.try_emplace(key, result);
            return result;
        }
    };

    for (int i = 0; i < pc.size(); ++i) {
        for (int j = i + 1; j < pc.size(); ++j) {
            if (pc[i].second == pc[j].second) {
                continue;
            }
            const string& aa_des_i = aa_des.at(aa.at(real_index.at(pc[i].first)));
            const string& aa_des_j = aa_des.at(aa.at(real_index.at(pc[j].first)));
            if (!check_dist_cache(pc[i].first, pc[j].first, aa_des_i, aa_des_j)) {
                continue;
            }
            for (int k = j + 1; k < pc.size(); ++k) {
                if ((pc[j].second == pc[k].second) || (pc[k].second == pc[i].second)) {
                    continue;
                }
                const string& aa_des_k = aa_des.at(aa.at(real_index.at(pc[k].first)));
                // set<int> distinct_classes = { pc[i].second, pc[j].second, pc[k].second };
                // if (distinct_classes.size() < 3) {
                //     continue;
                // }
                if (!check_dist_cache(pc[j].first, pc[k].first, aa_des_j, aa_des_k)) {
                    continue;
                }
                if (!check_dist_cache(pc[k].first, pc[i].first, aa_des_k, aa_des_i)) {
                    continue;
                }
                valid_combinations.push_back({pc[i], pc[j], pc[k]});
            }
        }
    }
    // }
    return valid_combinations;
}

tuple<Eigen::Vector3d, Eigen::Matrix3d> estimate_alignment(
    const vector<Eigen::Vector3d>& points1, 
    const vector<Eigen::Vector3d>& points2) {
    // Center the point sets
    Eigen::Vector3d centroid1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid2 = Eigen::Vector3d::Zero();

    for (const auto& point : points1) {
        centroid1 += point;
    }
    for (const auto& point : points2) {
        centroid2 += point;
    }

    centroid1 /= points1.size();
    centroid2 /= points2.size();

    vector<Eigen::Vector3d> centered_points1(points1.size());
    vector<Eigen::Vector3d> centered_points2(points2.size());

    for (int i = 0; i < points1.size(); ++i) {
        centered_points1[i] = points1[i] - centroid1;
        centered_points2[i] = points2[i] - centroid2;
    }

    // Compute the covariance matrix
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (int i = 0; i < points1.size(); ++i) {
        covariance_matrix += centered_points2[i] * centered_points1[i].transpose();
    }

    // Perform singular value decomposition (SVD)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d vh = svd.matrixV().transpose();

    // Compute the rotation matrix
    Eigen::Matrix3d rotation = vh.transpose() * u.transpose();

    // Compute the translation vector
    Eigen::Vector3d translation = centroid2 - rotation.transpose() * centroid1;

    return make_tuple(translation, rotation);
}

tuple<Eigen::Vector3d, Eigen::Matrix3d> euclidean_transformation_fit(
    const vector<Eigen::Vector3d>& protein, 
    const vector<Eigen::Vector3d>& protein_cb, 
    const vector<Eigen::Vector3d>& cavity, 
    const vector<Eigen::Vector3d>& cavity_cb, 
    const array<pair<int, int>, 3>& paa) {
    vector<Eigen::Vector3d> p;
    vector<Eigen::Vector3d> c;
    for (const auto& comb: paa) {
        p.push_back(protein[comb.first]);
        c.push_back(cavity[comb.second]);
        if (protein_cb[comb.first][0] != -10000000 && cavity_cb[comb.second][0] != -10000000) {
            p.push_back(protein_cb[comb.first]);
            c.push_back(cavity_cb[comb.second]);
        }
    }
    return estimate_alignment(p, c);
}

tuple<vector<double>, vector<pair<int, int>>> calculate_distances(
    const vector<Eigen::Vector3d>& transf, 
    const vector<Eigen::Vector3d>& cavity,
    const double threshold) {
    // vector<double> dist;
    // vector<pair<int, int>> indices;
    // for (int t = 0; t < cavity.size(); ++t) {
    //     for (int c = 0; c < transf.size(); ++c) {
    //         double distance = (transf[c] - cavity[t]).norm();
    //         if (distance < threshold) {
    //             dist.push_back(distance);
    //             indices.push_back({ c, t });
    //         }
    //     }
    // }
    // return make_tuple(dist, indices);
    vector<double> final_distances;
    vector<pair<int, int>> final_indices;
    PointCloudAdapter adapter{transf};
    KDTree tree(3, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();
    for (int i = 0; i < cavity.size(); ++i) {
        size_t nearest_index;
        double distance_squared;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&nearest_index, &distance_squared);
        tree.findNeighbors(resultSet, cavity[i].data(), nanoflann::SearchParameters(0));
        double distance = sqrt(distance_squared);
        if (distance < threshold) {
            final_distances.push_back(distance);
            final_indices.push_back(make_pair(static_cast<int>(nearest_index), i));
        }
    }
    return make_tuple(final_distances, final_indices);
}

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
    const double threshold) {
    array<pair<int, int>, 3> mapsel;
    if (i == -1) {
        // Seed the random number generator (only once)
        std::srand(std::time(0));
        vector<array<pair<int, int>, 3>> sampled;
        std::sample(valid_combinations.begin(), valid_combinations.end(), std::back_inserter(sampled), 1, std::mt19937{ std::random_device{}() });
        mapsel = sampled[0];
    }
    else {
        mapsel = valid_combinations[i];
    }

    auto [translation_vector, rotation] = euclidean_transformation_fit(protein, protein_cb, cavity_coords_used, cavity_coords_cb_used, mapsel);

    vector<Eigen::Vector3d> t_transformed(protein.size());
    for (int k = 0; k < protein.size(); ++k) {
        t_transformed[k] = rotation.transpose() * protein[k] + translation_vector;
    }
    auto [dist, indices] = calculate_distances(t_transformed, cavity, threshold);

    double sum_dist = 0.0;
    vector<pair<int, int>>mapping;
    vector<double> distances;
    for (int e = 0; e < cavity.size(); ++e) {
        vector<double> ppp;
        vector<pair<int, int>> indppp;
        for (int j = 0; j < indices.size();++j) {
            const auto& ind = indices[j];

            // Ensure new mappings based on distances are still with compatible amino acids
            if (ind.second == e && aa_des.at(aa.at(real_index.at(ind.first))) == aa_des.at(aa_cav.at(active[e]))) {
                ppp.push_back(dist[j]);
                indppp.push_back(ind);
            }
        }
        if (!ppp.empty()) {
            auto minindex = std::distance(ppp.begin(), std::min_element(ppp.begin(), ppp.end()));
            auto minmapindex = indppp[minindex];
            distances.push_back(ppp[minindex]);
            sum_dist += ppp[minindex];
            mapping.push_back(minmapindex);
        }
        if (mapping.empty()) {
            sum_dist = 100000.0;
        }
    }
    return make_tuple(mapping, sum_dist, distances, t_transformed, translation_vector, rotation);
}

tuple<double, double> get_distance_around(
    const vector<Eigen::Vector3d>& new_protein_coords,
    const vector<Eigen::Vector3d>& seed_coords,
    const vector<pair<int, int>> solution,
    const unordered_map<int, int>& real_index_seed,
    const vector<int>& active,
    const double aa_surrounding) {
    vector<double> distances;
    for (const auto& aamap : solution) {
        int aa_protein = aamap.first;
        int aa_cavity = real_index_seed.at(active[aamap.second]);
        for (int i = 1; i < aa_surrounding; ++i) { // Bug Fix: cannot use size_t type
            if (aa_protein - i > 0 && aa_cavity - i > 0) {
                double distance = (new_protein_coords[aa_protein - i] - seed_coords[aa_cavity - i]).norm();
                distances.push_back(distance);
            }
            if (aa_protein + i < new_protein_coords.size() && aa_cavity + i < seed_coords.size()) {
                double distance = (new_protein_coords[aa_protein + i] - seed_coords[aa_cavity + i]).norm();
                distances.push_back(distance);
            }
        }
    }
    double sum_distances = accumulate(distances.begin(), distances.end(), 0.0);
    return make_tuple(sum_distances, sum_distances / distances.size());
}

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
    const double threshold) {
    double final_dist = 100000000;
    vector<pair<int, int>> solution;
    vector<double> distances_selected;
    double distances_around_f;
    Eigen::Vector3d translation_vector_f;
    Eigen::Matrix3d rotation_f;
    vector<Eigen::Vector3d> t_transformed_f;
    // If the number of valid combinations is less than the number of iterations, the algorithm test each combination one by one, if not, it takes the combination randomly
    for (int i = 0; i < std::min(iterations, int(valid_combinations.size())); ++i) {
        try {
            auto [mapping, sum_dist, distances, t_transformed, translation_vector, rotation] = (iterations >= valid_combinations.size())
                ? calculate_final_distance(cavity, protein, protein_cb, valid_combinations, i, aa_des, aa, real_index, aa_cav, active, cavity_coords_used, cavity_coords_cb_used, threshold)
                : calculate_final_distance(cavity, protein, protein_cb, valid_combinations, -1, aa_des, aa, real_index, aa_cav, active, cavity_coords_used, cavity_coords_cb_used, threshold);
            // Select the mappings with less distances.there are few cases where the protein has 2 or more possible mappings and the one with less distance is not the right one
            unordered_set<int> mpn;
            for (const auto& mp : mapping) {
                mpn.insert(mp.first);
            }
            auto [dist_around_structure, distances_arround] = get_distance_around(t_transformed, seed_coords, mapping, real_index_seed, active, aa_around);
            sum_dist = sum_dist + dist_around_structure;
            if (sum_dist / mapping.size() <= final_dist) {
                final_dist = sum_dist / mapping.size();
                solution = mapping;
                distances_selected = distances;
                distances_around_f = distances_arround;
                translation_vector_f = translation_vector;
                rotation_f = rotation;
                t_transformed_f = t_transformed;
                if (final_dist == 0 && mpn.size() == cavity.size()) {
                    return { final_dist, distances_selected, distances_around_f, t_transformed_f, solution, translation_vector_f, rotation_f };
                }
            }
        }
        catch (const exception& e) {
            // cerr << "singular matrix?" << endl;
            // throw;
            cerr << e.what() << endl;
        }
    }
    return make_tuple(final_dist, distances_selected, distances_around_f, t_transformed_f, solution, translation_vector_f, rotation_f);
}

double calculate_score(vector<int>& indices, double weight) {
    int chunk_count = 0;
    int current_chunk = 1;

    for (int i = 1; i < indices.size(); ++i) {
        if (indices[i] == indices[i - 1] + 1) ++current_chunk;
        if (current_chunk > weight) {
            ++chunk_count;
            current_chunk = 1;
        }
    }
    if (current_chunk > weight) ++chunk_count;

    return chunk_count * weight;
}

double score_vector(vector<int>& indices) {
    constexpr double scores_weights[] = {5.0, 10.0, 15.0, 20.0};
    vector<double> scores;
    for (auto& weight : scores_weights) {
        scores.push_back(calculate_score(indices, weight));
    }

    double avgval = 0.0;
    double sum_weights = 0.0;
    for (int i = 0; i < 4; ++i) {
        avgval += scores[i] * scores_weights[i];
        sum_weights += scores_weights[i];
    }

    return avgval /= sum_weights;
}

tuple<vector<Eigen::Vector3d>, vector<Eigen::Vector3d>, vector<double>, vector<int>> find_nearest_neighbors(
    const vector<Eigen::Vector3d>& vector1,
    const vector<Eigen::Vector3d>& vector2
) {
    vector<Eigen::Vector3d> nn1;
    vector<Eigen::Vector3d> nn2;
    vector<double> distances;
    vector<int> indices;
    unordered_set<tuple<int, int, int>, tuple_hash> assigned;
    int index = 0;
    for (const auto& coord1: vector1) {
        double min_dist = numeric_limits<double>::infinity();
        const Eigen::Vector3d* nearest_coord = nullptr;
        for (const auto& coord2: vector2) {
            if (assigned.find(make_tuple(static_cast<int>(coord2.x() * 1e6), static_cast<int>(coord2.y() * 1e6), static_cast<int>(coord2.z() * 1e6))) != assigned.end()) {
                continue;
            }
            double dist = (coord1 - coord2).norm();
            if (dist < min_dist) {
                min_dist = dist;
                nearest_coord = &coord2;
            }
        }
        if (nearest_coord != nullptr && min_dist < 2.0) {
            nn1.push_back(coord1);
            nn2.push_back(*nearest_coord);
            assigned.insert(make_tuple(static_cast<int>(nearest_coord->x() * 1e6), static_cast<int>(nearest_coord->y() * 1e6), static_cast<int>(nearest_coord->z() * 1e6)));
            distances.push_back(min_dist);
            indices.push_back(index);
        }
        index += 1;
    }
    return make_tuple(nn1, nn2, distances, indices);
}

tuple<double, double, double> get_global_distance(
    const vector<Eigen::Vector3d>& coords1,
    const vector<Eigen::Vector3d>& coords2
) {
    try {
        vector<double> distances;
        vector<int> indices;
        constexpr double radius = 2.0;
        double radius_sq = radius * radius;

        int m = static_cast<int>(coords1.size());
        int n = static_cast<int>(coords2.size());
        vector<vector<double>> distance_matrix(m, vector<double>(n, 0.0));
        vector<vector<int>> score_matrix(m, vector<int>(n, 0));
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        // Build KDTree on coords1 (transformed protein coords)
        PointCloudAdapter adapter{coords1};
        KDTree tree(3, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();

        for (int k = 0; k < n; ++k) {
            std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
            tree.radiusSearch(coords2[k].data(), radius_sq, matches, nanoflann::SearchParameters());

            for (const auto& match : matches) {
                int idx = static_cast<int>(match.first);
                double dist = sqrt(match.second);
                score_matrix[idx][k] = 1;
                distance_matrix[idx][k] = dist;
                distances.push_back(dist);
                indices.push_back(idx);
            }
        }

        // DP alignment (Needleman-Wunsch style)
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                int current_score = score_matrix[i - 1][j - 1];
                int diagonal = dp[i - 1][j - 1] + current_score;
                int up = dp[i - 1][j];
                int left = dp[i][j - 1];
                dp[i][j] = std::max({diagonal, up, left});
            }
        }

        // Traceback
        vector<pair<int, int>> local_alignment;
        vector<double> local_distances;

        int i = m;
        int j = n;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + score_matrix[i - 1][j - 1]) {
                if (score_matrix[i - 1][j - 1] == 1) {
                    local_alignment.push_back({i - 1, j - 1});
                    local_distances.push_back(distance_matrix[i - 1][j - 1]);
                }
                --i; --j;
            } else if (i > 0 && dp[i][j] == dp[i - 1][j]) {
                --i;
            } else {
                --j;
            }
        }

        std::reverse(local_alignment.begin(), local_alignment.end());
        std::reverse(local_distances.begin(), local_distances.end());

        if (local_alignment.size() == 0) {
            return make_tuple(100.0, 100.0, 100.0);
        }

        double alignment_score = score_vector(indices);

        double rmsd = 0.0;
        for (const auto& val : distances) {
            rmsd += val * val;
        }
        rmsd = sqrt(rmsd / static_cast<double>(distances.size()));

        double percentage = static_cast<double>(indices.size()) / static_cast<double>(coords1.size());
        double rmsd_div_percentage = rmsd / percentage;

        return make_tuple(alignment_score, rmsd_div_percentage, percentage);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return make_tuple(40.0, 100.0, 100.0);
    }
}

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
    const unordered_map<int, int>& real_index_seed) {
    vector<int> active_used;
    vector<Eigen::Vector3d> cavity_coords_used;
    vector<Eigen::Vector3d> cavity_coords_cb_used;
    for (int x : index_used) {
        if (x < active.size() && x < cavity_coords.size() && x < cavity_coords_cb.size()) {
            active_used.push_back(active[x]);
            cavity_coords_used.push_back(cavity_coords[x]);
            cavity_coords_cb_used.push_back(cavity_coords_cb[x]);
        }
        else {
            return make_tuple(false, "", "", nan(""), nan(""), nan(""), nan(""), nan(""), "");
        }
    }

    double threshold = 3.0;
    vector<array<pair<int, int>, 3>> valid_combinations;
    {   // Create an object of the class Active_site where all the information is added
        unique_ptr<Active_site> active_site = make_unique<Active_site>(active_used, aa_cav, cavity_coords_used, aa_des);
        active_site->get_AA_groups(false);

        // Produce a vector with all the possible amino acid correspondences based on the aa_des dictionary
        vector<pair<int, int>> pc = active_site->get_possible_correspondences(protein_coords, aa, real_index);

        // Calculate the valid combinations by removing all the combinations of 3 amino acid mapping that are not in the appropiated distances with a threshold of 3 (that coud be an argument for the main function to be defined by the user)
        valid_combinations = active_site->get_all_possible_combinations(pc, protein_coords, aa, real_index, threshold, aa_des);
    }

    if (valid_combinations.size() > 0) {
        // The Ransac function
        auto [final_dist, distances, distances_arround, t_transformed, solution, translation_vector, rotation] = ransac_protein(
            cavity_coords, protein_coords, protein_coords_cb, seed_coords, valid_combinations,
            iterations, aa_des, aa, real_index, real_index_seed, aa_cav, active, cavity_coords_used, cavity_coords_cb_used, 3.0, threshold);

        // It only saves the solution if the maximum deviating distance is less than 1 (that could also be another argument in the main function defined by the user, or use the same threshold as in the previous function)
        if (accumulate(distances.begin(), distances.end(), 0.0) / distances.size() < 2 && distances_arround < 3.0 && solution.size() >= 3) {
            auto [rmsd, minrmsd, percentage] = get_global_distance(t_transformed, seed_coords);
            string strmapping;
            for (const auto& aamap : solution) {
                string aa_protein = aa.at(real_index.at(aamap.first));
                string aa_cavity = aa_cav.at(active[aamap.second]);
                strmapping += to_string(real_index.at(aamap.first)) + aa_protein + "-" + to_string(active[aamap.second]) + aa_cavity + ",";
            }
            string str_distances = accumulate(distances.begin() + 1, distances.end(), to_string(distances[0]), [](const std::string& a, double b) { return a + "," + to_string(b); });
            return make_tuple(true, case_protein_name, strmapping, accumulate(distances.begin(), distances.end(), 0.0) / active.size(), distances_arround, rmsd, minrmsd, percentage, str_distances);
        }
        else {
            return make_tuple(false, "", "", nan(""), nan(""), nan(""), nan(""), nan(""), "");
        }
    }
    else {
        return make_tuple(false, "", "", nan(""), nan(""), nan(""), nan(""), nan(""), "");
    }
}

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
    const unordered_map<int, int>& real_index_seed) {
    auto result = proteinRansacMain(index_used, aa_des, iterations, case_protein + "-" + seed_protein, protein_coords, protein_coords_cb, seed_coords, cavity_coords, cavity_coords_cb,
        aa, aa_cav, active, real_index, real_index_seed);
    if (get<0>(result)) {
        return make_tuple(seed_protein + " -- " + ec, static_cast<int>(seed_coords.size()), static_cast<int>(protein_coords.size()), get<2>(result), get<8>(result), get<3>(result), get<4>(result), get<5>(result), get<6>(result), get<7>(result), ec.substr(0, ec.find('-')));
    }
    else {
        return make_tuple("", static_cast<int>(nan("")), static_cast<int>(nan("")), "", "", nan(""), nan(""), nan(""), nan(""), nan(""), "");
    }
}

vector<result_t> concurrentMain(
    const profile_t& profile,
    const string& segment_name,
    const string& table_name) {
    try {
        ThreadPool pool(get_thread_pool_size());
        bip::managed_shared_memory shared_entries(bip::open_read_only, segment_name.c_str());
        ShmemEntryVector* entries = shared_entries.find<ShmemEntryVector>(table_name.c_str()).first;
        vector<future<result_t>> futures;
        vector<result_t> results;
        auto [case_protein, protein_coords, protein_coords_cb, aa, real_index] = profile;
        for (int i = 0; i < entries->size(); ++i) {
            // auto [active_indices, seed_protein, ec, seed_coords, cavity_coords, cavity_coords_cb, aa_cav, active, real_index_seed] = (*entries)[i];
            const auto& entry = (*entries)[i];
            vector<int> active_indices(entry.active_indices.begin(), entry.active_indices.end());
            string seed_protein = string(entry.seed_protein.begin(), entry.seed_protein.end());
            string ec = string(entry.ec.begin(), entry.ec.end());
            vector<Eigen::Vector3d> seed_coords(entry.seed_coords.begin(), entry.seed_coords.end());
            vector<Eigen::Vector3d> cavity_coords(entry.cavity_coords.begin(), entry.cavity_coords.end());
            vector<Eigen::Vector3d> cavity_coords_cb(entry.cavity_coords_cb.begin(), entry.cavity_coords_cb.end());
            unordered_map<int, string> aa_cav;
            aa_cav.reserve(entry.aa_cav.size());
            for (const auto& [k, v] : entry.aa_cav) {
                // auto it = aa_names.find(v);
                // if (it != aa_names.end()) {
                //     aa_cav.emplace(k, it->second);
                // }
                aa_cav.try_emplace(k, aa_names.at(v));
            }
            vector<int> active(entry.active.begin(), entry.active.end());
            unordered_map<int, int> real_index_seed = entry.real_index_seed;

            futures.push_back(pool.enqueue(
                proteinRansacMain_t,
                i,
                std::move(active_indices),
                std::cref(aa_des),
                3000,
                std::move(seed_protein),
                std::cref(case_protein),
                std::move(ec),
                std::cref(protein_coords),
                std::cref(protein_coords_cb),
                std::move(seed_coords),
                std::move(cavity_coords),
                std::move(cavity_coords_cb),
                std::cref(aa),
                std::move(aa_cav),
                std::move(active),
                std::cref(real_index),
                std::move(real_index_seed)
            ));
        }
        for (auto& fut : futures) {
            auto result = fut.get();
            if (get<0>(result) != "") {
                results.push_back(result);
            }
        }
        return results;
    } catch (const exception& e) {
        cerr << e.what() << endl;
        throw;
    }
}

PYBIND11_MODULE(ActSeekLib, m) {
    m.def("concurrentMain", &concurrentMain, "concurrentMain");
    m.def("createSharedEntries", &createSharedEntries, "createSharedEntries");
    m.def("destroySharedEntries", &destroySharedEntries, "destroySharedEntries");
    m.def("get_thread_pool_size", &get_thread_pool_size, "Return ActSeek C++ thread pool size");
    m.def("set_thread_pool_size", &set_thread_pool_size, "Set ActSeek C++ thread pool size");
    m.def("set_gpu_enabled", &set_gpu_enabled, "Enable or disable GPU usage");
    m.def("gpu_enabled", &gpu_enabled, "Return whether GPU usage is enabled");
}

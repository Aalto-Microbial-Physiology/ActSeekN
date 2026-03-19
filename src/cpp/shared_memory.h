#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

#include "ActSeekLib.h"

namespace bip = boost::interprocess;
using namespace std;

using ShmemAllocatorString = bip::allocator<char, bip::managed_shared_memory::segment_manager>;
using ShmemAllocatorInt = bip::allocator<int, bip::managed_shared_memory::segment_manager>;
using ShmemAllocator3d = bip::allocator<Eigen::Vector3d, bip::managed_shared_memory::segment_manager>;
// using ShmemAllocatorMapIntString = bip::allocator<std::pair<const int, bip::basic_string<char, std::char_traits<char>, ShmemAllocatorString>>, bip::managed_shared_memory::segment_manager>;

using ShmemString = bip::basic_string<char, std::char_traits<char>, ShmemAllocatorString>;
using ShmemVector3d = bip::vector<Eigen::Vector3d, ShmemAllocator3d>;
using ShmemVectorInt = bip::vector<int, ShmemAllocatorInt>;
// using ShmemMapIntString = std::unordered_map<int, ShmemString, std::hash<int>, std::equal_to<int>, ShmemAllocatorMapIntString>;

using entry_t = tuple<vector<int>, string, string, vector<Eigen::Vector3d>, vector<Eigen::Vector3d>, vector<Eigen::Vector3d>,
    unordered_map<int, int>, vector<int>, unordered_map<int, int>>;
// using shmem_entry_t = std::tuple<ShmemVectorInt, ShmemString, ShmemString, 
//                            ShmemVector3d, ShmemVector3d, ShmemVector3d, 
//                            ShmemMapIntString, ShmemVectorInt, std::unordered_map<int, int>>;
struct ShmemEntry;
using ShmemAllocatorEntry = bip::allocator<ShmemEntry, bip::managed_shared_memory::segment_manager>;
struct ShmemEntry {
    ShmemVectorInt active_indices;
    ShmemString seed_protein;
    ShmemString ec;
    ShmemVector3d seed_coords;
    ShmemVector3d cavity_coords;
    ShmemVector3d cavity_coords_cb;
    std::unordered_map<int, int> aa_cav;
    ShmemVectorInt active;
    std::unordered_map<int, int> real_index_seed;

    ShmemEntry(const ShmemVectorInt& active_indices,
               const ShmemString& seed_protein,
               const ShmemString& ec,
               const ShmemVector3d& seed_coords,
               const ShmemVector3d& cavity_coords,
               const ShmemVector3d& cavity_coords_cb,
               const std::unordered_map<int, int> & aa_cav,
               const ShmemVectorInt& active,
               const std::unordered_map<int, int>& real_index_seed,
               const ShmemAllocatorEntry& alloc)
        : active_indices(active_indices, alloc),
          seed_protein(seed_protein, alloc),
          ec(ec, alloc),
          seed_coords(seed_coords.begin(), seed_coords.end(), alloc),
          cavity_coords(cavity_coords.begin(), cavity_coords.end(), alloc),
          cavity_coords_cb(cavity_coords_cb.begin(), cavity_coords_cb.end(), alloc),
          aa_cav(aa_cav),
          active(active.begin(), active.end(), alloc),
          real_index_seed(real_index_seed) {
        // for (const auto& [key, value] : aa_cav) {
        //     this->aa_cav.emplace(key, ShmemString(value.c_str(), alloc));
        // }
    }
};
using ShmemEntryVector = bip::vector<ShmemEntry, ShmemAllocatorEntry>;

size_t estimate_dynamic_memory(const entry_t& entry) {
    size_t total_dynamic_memory = 0;

    const auto& [active_indices, seed_protein, ec, seed_coords, cavity_coords, cavity_coords_cb, aa_cav, active, real_index_seed] = entry;

    total_dynamic_memory += active_indices.size() * sizeof(int);
    total_dynamic_memory += active.size() * sizeof(int);

    total_dynamic_memory += seed_protein.capacity();
    total_dynamic_memory += ec.capacity();

    total_dynamic_memory += seed_coords.size() * sizeof(Eigen::Vector3d);
    total_dynamic_memory += cavity_coords.size() * sizeof(Eigen::Vector3d);
    total_dynamic_memory += cavity_coords_cb.size() * sizeof(Eigen::Vector3d);

    // for (const auto& pair : aa_cav) {
    //     total_dynamic_memory += sizeof(int) + pair.second.capacity();
    // }
    total_dynamic_memory += aa_cav.size() * (sizeof(int) + sizeof(int));
    total_dynamic_memory += real_index_seed.size() * (sizeof(int) + sizeof(int));

    return total_dynamic_memory;
}

string createSharedEntries(const vector<entry_t>& entries, const string& table_name, const int pid) {
    size_t total_size = 0;
    for (const auto& entry: entries) {
        total_size += sizeof(entry_t);
        total_size += estimate_dynamic_memory(entry);
    }
    total_size = static_cast<size_t>(total_size * 2);

    // Create a shared memory segment
    // bip::permissions perms;
    // perms.set_unrestricted();
    string segment_name = "shmem_" + std::to_string(pid);;
    bip::managed_shared_memory segment(bip::create_only, segment_name.c_str(), total_size);

    // Construct the allocator for the shared memory
    const ShmemAllocatorEntry alloc_inst(segment.get_segment_manager());

    // Construct a vector of entries in shared memory
    ShmemEntryVector *shmem_entries = segment.find_or_construct<ShmemEntryVector>(table_name.c_str())(alloc_inst);

    for (const auto& entry: entries) {
        const auto& [active_indices, seed_protein, ec, seed_coords, cavity_coords, cavity_coords_cb, aa_cav, active, real_index_seed] = entry;
        ShmemVectorInt shm_active_indices(active_indices.begin(), active_indices.end(), alloc_inst);
        ShmemString shm_seed_protein(seed_protein.c_str(), alloc_inst);
        ShmemString shm_ec(ec.c_str(), alloc_inst);
        ShmemVector3d shm_seed_coords(seed_coords.begin(), seed_coords.end(), alloc_inst);
        ShmemVector3d shm_cavity_coords(cavity_coords.begin(), cavity_coords.end(), alloc_inst);
        ShmemVector3d shm_cavity_coords_cb(cavity_coords_cb.begin(), cavity_coords_cb.end(), alloc_inst);
        // ShmemMapIntString shm_aa_cav(alloc_inst);
        // for (const auto& [aa, cav] : aa_cav) {
        //     shm_aa_cav.emplace(aa, ShmemString(cav.c_str(), alloc_inst));
        // }
        ShmemVectorInt shm_active(active.begin(), active.end(), alloc_inst);
        ShmemEntry shmem_entry(shm_active_indices, shm_seed_protein, shm_ec, shm_seed_coords, shm_cavity_coords, shm_cavity_coords_cb, aa_cav, shm_active, real_index_seed, alloc_inst);
        shmem_entries->push_back(shmem_entry);
    }

    return segment_name;
}

void destroySharedEntries(const string& segment_name) {
    boost::interprocess::shared_memory_object::remove(segment_name.c_str());
}


#endif
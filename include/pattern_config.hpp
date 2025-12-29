#ifndef PATTERN_CONFIG_H
#define PATTERN_CONFIG_H

#include <string>
#include <map>
#include <set>
#include <iostream>

// Enums for configuration
enum class ThreadRole : uint8_t {
    INACTIVE = 0,
    WRITER = 1,
    READER = 2,
    DUMMY_READER = 3,
    DUMMY_WRITER = 4
};

enum class ThreadScope : uint8_t {
    THREAD = 0,
    BLOCK = 1,
    DEVICE = 2,
    SYSTEM = 3
};

enum class MemoryOrdering : uint8_t {
    RELAXED = 0,
    ACQUIRE = 1,
    RELEASE = 2,
    ACQ_REL = 3
};

// Helper functions for string conversion
inline const char* role_to_string(ThreadRole role) {
    switch(role) {
        case ThreadRole::INACTIVE: return "inactive";
        case ThreadRole::WRITER: return "writer";
        case ThreadRole::READER: return "reader";
        case ThreadRole::DUMMY_READER: return "dummy_reader";
        case ThreadRole::DUMMY_WRITER: return "dummy_writer";
        default: return "unknown";
    }
}

inline const char* scope_to_string(ThreadScope scope) {
    switch(scope) {
        case ThreadScope::THREAD: return "thread";
        case ThreadScope::BLOCK: return "block";
        case ThreadScope::DEVICE: return "device";
        case ThreadScope::SYSTEM: return "system";
        default: return "unknown";
    }
}

inline const char* ordering_to_string(MemoryOrdering ordering) {
    switch(ordering) {
        case MemoryOrdering::RELAXED: return "relaxed";
        case MemoryOrdering::ACQUIRE: return "acquire";
        case MemoryOrdering::RELEASE: return "release";
        case MemoryOrdering::ACQ_REL: return "acq_rel";
        default: return "unknown";
    }
}

inline ThreadRole string_to_role(const std::string& str) {
    if (str == "writer") return ThreadRole::WRITER;
    if (str == "reader") return ThreadRole::READER;
    if (str == "dummy_reader") return ThreadRole::DUMMY_READER;
    if (str == "dummy_writer") return ThreadRole::DUMMY_WRITER;
    if (str == "inactive") return ThreadRole::INACTIVE;
    return ThreadRole::INACTIVE;
}

inline ThreadScope string_to_scope(const std::string& str) {
    if (str == "thread") return ThreadScope::THREAD;
    if (str == "block") return ThreadScope::BLOCK;
    if (str == "device") return ThreadScope::DEVICE;
    if (str == "system") return ThreadScope::SYSTEM;
    return ThreadScope::DEVICE;
}

inline MemoryOrdering string_to_ordering(const std::string& str) {
    if (str == "relaxed") return MemoryOrdering::RELAXED;
    if (str == "acquire") return MemoryOrdering::ACQUIRE;
    if (str == "release") return MemoryOrdering::RELEASE;
    if (str == "acq_rel") return MemoryOrdering::ACQ_REL;
    return MemoryOrdering::RELAXED;
}

// Per-thread configuration (compact: 4 bytes)
struct ThreadConfig {
    ThreadRole role;
    ThreadScope scope;
    MemoryOrdering ordering;
    ThreadScope watch_flag;  // Which flag scope to observe
    bool caching;            // Whether to use caching variant
};

// Pattern storage
struct PatternConfig {
    std::string name;
    std::string description;
    bool multi_writer;  // If true, use 4 scope-specific buffers
    
    // GPU: [block_id][thread_id]
    ThreadConfig gpu_threads[8][64];
    
    // CPU: [core_id]
    ThreadConfig cpu_threads[32];
    
    int gpu_num_blocks;
    int gpu_threads_per_block;
    int cpu_num_threads;
    
    PatternConfig() 
        : name("")
        , description("")
        , multi_writer(false)
        , gpu_num_blocks(8)
        , gpu_threads_per_block(64)
        , cpu_num_threads(32) {}
};

// Pattern registry
class PatternRegistry {
private:
    std::map<std::string, PatternConfig> patterns_;
    
public:
    bool load_from_yaml(const std::string& yaml_path);
    const PatternConfig* get_pattern(const std::string& name) const;
    void list_patterns() const;
    bool validate_pattern(const PatternConfig& pat);
};

// Global instance
extern PatternRegistry g_pattern_registry;

#endif // PATTERN_CONFIG_H

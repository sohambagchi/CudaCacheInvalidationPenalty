#include "pattern_config.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

// Global instance
PatternRegistry g_pattern_registry;

// Helper to trim whitespace
static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// Helper to parse value after colon
static std::string parse_value(const std::string& line) {
    size_t colon = line.find(':');
    if (colon == std::string::npos) return "";
    
    std::string value = line.substr(colon + 1);
    value = trim(value);
    
    // Remove quotes if present
    if (!value.empty() && (value.front() == '"' || value.front() == '\'')) {
        value = value.substr(1);
    }
    if (!value.empty() && (value.back() == '"' || value.back() == '\'')) {
        value = value.substr(0, value.length() - 1);
    }
    
    return value;
}

// Helper to get indentation level
static int get_indent(const std::string& line) {
    int indent = 0;
    for (char c : line) {
        if (c == ' ') indent++;
        else if (c == '\t') indent += 4;
        else break;
    }
    return indent;
}

// Helper to parse thread range (e.g., "thread_0", "threads_1_63", "all_threads")
static void parse_thread_range(const std::string& key, int& start, int& end, int max_threads) {
    if (key == "all_threads") {
        start = 0;
        end = max_threads - 1;
    } else if (key.find("threads_") == 0) {
        // threads_1_63
        size_t underscore1 = key.find('_', 8);
        if (underscore1 != std::string::npos) {
            start = std::stoi(key.substr(8, underscore1 - 8));
            end = std::stoi(key.substr(underscore1 + 1));
        }
    } else if (key.find("thread_") == 0) {
        // thread_0
        start = end = std::stoi(key.substr(7));
    }
}

// Helper to parse block range (e.g., "block_0", "blocks_2_7")
static void parse_block_range(const std::string& key, int& start, int& end, int max_blocks) {
    if (key.find("blocks_") == 0) {
        // blocks_2_7
        size_t underscore1 = key.find('_', 7);
        if (underscore1 != std::string::npos) {
            start = std::stoi(key.substr(7, underscore1 - 7));
            end = std::stoi(key.substr(underscore1 + 1));
        }
    } else if (key.find("block_") == 0) {
        // block_0
        start = end = std::stoi(key.substr(6));
    }
}

// Helper to parse thread config from inline format: {role: reader, ordering: acquire, ...}
static ThreadConfig parse_inline_config(const std::string& value) {
    ThreadConfig cfg = {ThreadRole::INACTIVE, ThreadScope::DEVICE, MemoryOrdering::RELAXED, ThreadScope::DEVICE};
    
    // Remove braces
    std::string cleaned = value;
    if (cleaned.front() == '{') cleaned = cleaned.substr(1);
    if (cleaned.back() == '}') cleaned = cleaned.substr(0, cleaned.length() - 1);
    
    // Split by comma
    std::istringstream ss(cleaned);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        size_t colon = token.find(':');
        if (colon == std::string::npos) continue;
        
        std::string key = trim(token.substr(0, colon));
        std::string val = trim(token.substr(colon + 1));
        
        if (key == "role") {
            cfg.role = string_to_role(val);
        } else if (key == "ordering") {
            cfg.ordering = string_to_ordering(val);
        } else if (key == "scope") {
            cfg.scope = string_to_scope(val);
        } else if (key == "watch_flag") {
            cfg.watch_flag = string_to_scope(val);
        } else if (key == "caching") {
            cfg.caching = (val == "true" || val == "True" || val == "TRUE");
        }
    }
    
    return cfg;
}

bool PatternRegistry::validate_pattern(const PatternConfig& pat) {
    std::set<ThreadScope> required_scopes;
    std::map<ThreadScope, int> writer_count_per_scope;
    int total_writers = 0;

    // 1. Scan GPU Threads
    for (int b = 0; b < pat.gpu_num_blocks; b++) {
        for (int t = 0; t < pat.gpu_threads_per_block; t++) {
            const auto& cfg = pat.gpu_threads[b][t];
            if (cfg.role == ThreadRole::READER) {
                required_scopes.insert(cfg.watch_flag);
            }
            else if (cfg.role == ThreadRole::WRITER) {
                writer_count_per_scope[cfg.scope]++;
                total_writers++;
            }
        }
    }

    // 2. Scan CPU Threads
    for (int t = 0; t < pat.cpu_num_threads; t++) {
        const auto& cfg = pat.cpu_threads[t];
        if (cfg.role == ThreadRole::READER) {
            required_scopes.insert(cfg.watch_flag);
        }
        else if (cfg.role == ThreadRole::WRITER) {
            writer_count_per_scope[cfg.scope]++;
            total_writers++;
        }
    }

    // 3. Multi-writer mode validation
    if (pat.multi_writer) {
        if (total_writers != 4) {
            std::cerr << "[ERROR] Multi-writer mode requires exactly 4 writers, found " 
                      << total_writers << std::endl;
            return false;
        }
        
        // Must have exactly one writer per scope
        for (auto scope : {ThreadScope::THREAD, ThreadScope::BLOCK, ThreadScope::DEVICE, ThreadScope::SYSTEM}) {
            if (writer_count_per_scope[scope] != 1) {
                std::cerr << "[ERROR] Multi-writer mode requires exactly 1 writer for scope " 
                          << scope_to_string(scope) << ", found " 
                          << writer_count_per_scope[scope] << std::endl;
                return false;
            }
        }
        
        // 4. Verify Coverage (multi-writer only)
        // Every scope required by a reader must be provided by at least one writer
        for (const auto& scope : required_scopes) {
            if (writer_count_per_scope[scope] == 0) {
                std::cerr << "[ERROR] Pattern Invalid: Readers are waiting for scope " 
                          << scope_to_string(scope) 
                          << ", but no Writer is configured to signal that scope." << std::endl;
                return false;
            }
        }
    } else {
        // Single-writer mode: just need at least one writer (it sets all flags)
        if (total_writers == 0) {
            std::cerr << "[ERROR] Pattern Invalid: No writers found. At least one writer is required." << std::endl;
            return false;
        }
    }

    return true;
}

bool PatternRegistry::load_from_yaml(const std::string& yaml_path) {
    std::ifstream file(yaml_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open: " << yaml_path << std::endl;
        return false;
    }
    
    PatternConfig current_pattern;
    // Initialize all thread configs to INACTIVE
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 64; j++) {
            current_pattern.gpu_threads[i][j] = {ThreadRole::INACTIVE, ThreadScope::DEVICE, MemoryOrdering::RELAXED, ThreadScope::DEVICE};
        }
    }
    for (int i = 0; i < 32; i++) {
        current_pattern.cpu_threads[i] = {ThreadRole::INACTIVE, ThreadScope::DEVICE, MemoryOrdering::RELAXED, ThreadScope::DEVICE};
    }
    
    std::string line;
    int line_num = 0;
    bool in_pattern = false;
    bool in_gpu = false;
    bool in_cpu = false;
    bool in_blocks = false;
    int current_block_start = -1;
    int current_block_end = -1;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip empty lines and comments
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        
        int indent = get_indent(line);
        
        // Pattern start
        if (trimmed.find("- name:") == 0) {
            // Save previous pattern
            if (in_pattern && !current_pattern.name.empty()) {
                if (validate_pattern(current_pattern)) {
                    patterns_[current_pattern.name] = current_pattern;
                } else {
                    std::cerr << "[WARNING] Skipping invalid pattern: " << current_pattern.name << std::endl;
                }
            }
            
            // Start new pattern
            current_pattern = PatternConfig();
            // Initialize all thread configs to INACTIVE
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 64; j++) {
                    current_pattern.gpu_threads[i][j] = {ThreadRole::INACTIVE, ThreadScope::DEVICE, MemoryOrdering::RELAXED, ThreadScope::DEVICE};
                }
            }
            for (int i = 0; i < 32; i++) {
                current_pattern.cpu_threads[i] = {ThreadRole::INACTIVE, ThreadScope::DEVICE, MemoryOrdering::RELAXED, ThreadScope::DEVICE};
            }
            current_pattern.name = parse_value(trimmed);
            in_pattern = true;
            in_gpu = false;
            in_cpu = false;
            in_blocks = false;
        }
        else if (trimmed.find("description:") == 0 && in_pattern) {
            current_pattern.description = parse_value(trimmed);
        }
        else if (trimmed.find("multi_writer:") == 0 && in_pattern) {
            std::string val = parse_value(trimmed);
            current_pattern.multi_writer = (val == "true" || val == "True" || val == "TRUE");
        }
        else if (trimmed == "gpu:" && in_pattern) {
            in_gpu = true;
            in_cpu = false;
            in_blocks = false;
        }
        else if (trimmed == "cpu:" && in_pattern) {
            in_cpu = true;
            in_gpu = false;
            in_blocks = false;
        }
        else if (trimmed.find("num_blocks:") == 0 && in_gpu) {
            current_pattern.gpu_num_blocks = std::stoi(parse_value(trimmed));
        }
        else if (trimmed.find("threads_per_block:") == 0 && in_gpu) {
            current_pattern.gpu_threads_per_block = std::stoi(parse_value(trimmed));
        }
        else if (trimmed.find("num_threads:") == 0 && in_cpu) {
            current_pattern.cpu_num_threads = std::stoi(parse_value(trimmed));
        }
        else if (trimmed == "blocks:" && in_gpu) {
            in_blocks = true;
        }
        else if (trimmed == "threads:" && in_cpu) {
            // CPU threads section
        }
        else if (in_blocks && (trimmed.find("block_") == 0 || trimmed.find("blocks_") == 0) && trimmed.back() == ':') {
            // Block specification
            std::string block_key = trimmed.substr(0, trimmed.length() - 1);
            parse_block_range(block_key, current_block_start, current_block_end, current_pattern.gpu_num_blocks);
        }
        else if (in_blocks && current_block_start >= 0 && 
                 (trimmed.find("thread_") == 0 || trimmed.find("threads_") == 0 || trimmed.find("all_threads") == 0)) {
            // Thread specification
            size_t colon = trimmed.find(':');
            if (colon != std::string::npos) {
                std::string thread_key = trimmed.substr(0, colon);
                std::string config_str = trim(trimmed.substr(colon + 1));
                
                int thread_start, thread_end;
                parse_thread_range(thread_key, thread_start, thread_end, current_pattern.gpu_threads_per_block);
                
                ThreadConfig cfg = parse_inline_config(config_str);
                
                // Apply to all threads in range, all blocks in range
                for (int b = current_block_start; b <= current_block_end; b++) {
                    for (int t = thread_start; t <= thread_end; t++) {
                        current_pattern.gpu_threads[b][t] = cfg;
                    }
                }
            }
        }
        else if (in_cpu && !in_blocks && 
                 (trimmed.find("thread_") == 0 || trimmed.find("threads_") == 0 || trimmed.find("all_threads") == 0)) {
            // CPU thread specification
            size_t colon = trimmed.find(':');
            if (colon != std::string::npos) {
                std::string thread_key = trimmed.substr(0, colon);
                std::string config_str = trim(trimmed.substr(colon + 1));
                
                int thread_start, thread_end;
                parse_thread_range(thread_key, thread_start, thread_end, current_pattern.cpu_num_threads);
                
                ThreadConfig cfg = parse_inline_config(config_str);
                
                // Apply to all threads in range
                for (int t = thread_start; t <= thread_end; t++) {
                    current_pattern.cpu_threads[t] = cfg;
                }
            }
        }
    }
    
    // Save last pattern
    if (in_pattern && !current_pattern.name.empty()) {
        if (validate_pattern(current_pattern)) {
            patterns_[current_pattern.name] = current_pattern;
        } else {
            std::cerr << "[WARNING] Skipping invalid pattern: " << current_pattern.name << std::endl;
        }
    }
    
    std::cout << "[INFO] Loaded " << patterns_.size() << " patterns from " << yaml_path << std::endl;
    
    return !patterns_.empty();
}

const PatternConfig* PatternRegistry::get_pattern(const std::string& name) const {
    auto it = patterns_.find(name);
    if (it != patterns_.end()) {
        return &it->second;
    }
    return nullptr;
}

void PatternRegistry::list_patterns() const {
    std::cout << "\n=== Available Patterns ===" << std::endl;
    for (const auto& pair : patterns_) {
        std::cout << "  • " << pair.first << std::endl;
        if (!pair.second.description.empty()) {
            std::cout << "    " << pair.second.description << std::endl;
        }
    }
    std::cout << std::endl;
}

import re
import glob

# Enhanced regex pattern for CUDA and C++ function signatures
func_pattern = re.compile(
    r'^\s*(?P<template>template\s*<[^>]+>\s*)?'  # Matches 'template <typename R>' if present
    r'(?P<qualifiers>(?:static|inline|constexpr)?\s*(?:__global__|__device__|__host__)?\s*(?:static|inline|constexpr)?)\s*'  # Matches qualifiers
    r'(?P<return_type>[\w:\*<>]+)\s+'  # Matches return type
    r'(?P<func_name>\w+)\s*'  # Matches function name
    r'\((?P<params>[^)]*)\)'  # Matches function parameters
)


for filename in glob.glob("*.cu"):
    header_content = f"#ifndef {filename.replace('.', '_').replace('cu', 'cuh').upper()}\n#define {filename.replace('.', '_').replace('cu', 'cuh').upper()}\n\n"
    with open(filename, "r") as file:
        for line in file:
            # Remove __attribute__((...)) annotations
            line = re.sub(r'__attribute__\s*\(\([^)]*\)\).', '', line)

            match = func_pattern.match(line)
            
            if 'static' in line or 'inline' in line:
                print(line)
                print(match)
            
            if match:
                template = match.group("template") or ""
                qualifiers = match.group("qualifiers") or ""
                return_type = match.group("return_type").strip()
                func_name = match.group("func_name")
                params = match.group("params")

                # Ensure proper qualifiers
                if "__global__" in qualifiers:
                    qualifiers = f"__global__ {qualifiers.replace('__global__', '').strip()}"
                elif "__device__" in qualifiers:
                    qualifiers = f"__device__ {qualifiers.replace('__device__', '').strip()}"
                elif "__host__" in qualifiers or "static" in qualifiers or "inline" in qualifiers:
                    qualifiers = f"__host__ {qualifiers.replace('__host__', '').strip()}"

                header_content += f"{template}{qualifiers} {return_type} {func_name}({params});\n"

    header_content += f"\n#endif // {filename.replace('.', '_').replace('cu', 'cuh').upper()}\n"
    
    with open(filename.replace('.cu', '.cuh'), "w") as header_file:
        header_file.write(header_content)

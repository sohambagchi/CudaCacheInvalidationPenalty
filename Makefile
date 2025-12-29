OUTPUT = output/cache_invalidation_testing
SRC = src/cache_invalidation_testing.cu
PATTERN_SRC = src/pattern_config.cpp
HEADERS = include/types.hpp include/pattern_config.hpp include/pattern_dispatch.cuh include/pattern_dispatch_cpu.hpp
NVCC = nvcc
NVCC_FLAGS = -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87 -Iinclude
LIBS = -lnuma -lm

SIZES = DATA_SIZE_32 DATA_SIZE_64 DATA_SIZE_16 DATA_SIZE_8

all: $(foreach size,$(SIZES),$(OUTPUT)_$(size).out)

output:
	mkdir -p output

define make_target
$(OUTPUT)_$(1).out: $(SRC) $(PATTERN_SRC) $(HEADERS) | output
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -DPATTERN_DISPATCH -D$(1) -o $$@ $$(SRC) $$(PATTERN_SRC)
endef

$(foreach size,$(SIZES),$(eval $(call make_target,$(size))))

clean:
	rm -rf output
	rm -f *.out *.ptx



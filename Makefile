OUTPUT = cache_invalidation_testing
SRC = cache_invalidation_testing.cu
NVCC = nvcc
NVCC_FLAGS = -g -O0 -Xcompiler -O0 -Xcicc -O0 -arch=sm_87 
LIBS = -lnuma -lm

SCOPES = cuda\:\:thread_scope_system cuda\:\:thread_scope_thread cuda\:\:thread_scope_block cuda\:\:thread_scope_device
SIZES = uint8_t uint16_t uint32_t uint64_t


all: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_$(scope)_$(size).out))

define make_target
$(OUTPUT)_$(1)_$(2).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -DSCOPE=$(1) -DDATA_SIZE=$(2) -o $$@ $$(SRC)
endef

# $(TARGET): $(SRC)
# 	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC) $(LIBS)


$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target,$(scope),$(size)))))

clean:
	rm -f *.out
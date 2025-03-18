OUTPUT = cache_invalidation_testing
SRC = cache_invalidation_testing.cu
HEADERS = cache_invalidation_testing.cuh 
NVCC = nvcc
NVCC_FLAGS = -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87
LIBS = -lnuma -lm

# SCOPES = cuda\:\:thread_scope_system cuda\:\:thread_scope_thread cuda\:\:thread_scope_device # cuda\:\:thread_scope_block

SCOPES = CUDA_THREAD_SCOPE_THREAD CUDA_THREAD_SCOPE_DEVICE CUDA_THREAD_SCOPE_BLOCK # CUDA_THREAD_SCOPE_SYSTEM #
# SIZES = uint64_t uint32_t uint16_t uint8_t
SIZES = DATA_SIZE_32 #DATA_SIZE_64  DATA_SIZE_16 DATA_SIZE_8
BUFFER = BUFFER_SAME #BUFFER_DIFF
SIGNAL_SCOPES = SIGNAL_THREAD_SCOPE_DEVICE SIGNAL_THREAD_SCOPE_SYSTEM # SIGNAL_THREAD_SCOPE_THREAD SIGNAL_THREAD_SCOPE_BLOCK 


all: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(foreach signal,$(SIGNAL_SCOPES),$(OUTPUT)_$(scope)_$(size)_$(buf)_$(signal).out))))

define make_target
$(OUTPUT)_$(1)_$(2)_$(3)_$(4).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -D$(4) -o $$@ $$(SRC)
	# $$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -D$(4) -ptx $$(SRC)
	# mv $(OUTPUT).ptx $(OUTPUT)_$(1)_$(2)_$(3).ptx
endef

# $(TARGET): $(SRC)
# 	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC) $(LIBS)


$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(foreach signal,$(SIGNAL_SCOPES),$(eval $(call make_target,$(scope),$(size),$(buf),$(signal)))))))

clean:
	rm -f *.out *.ptx

OUTPUT = output/cache_invalidation_testing
SRC = cache_invalidation_testing.cu
HEADERS = gpu_kernels.cuh cpu_functions.h types.h
NVCC = nvcc
NVCC_FLAGS = -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87
LIBS = -lnuma -lm

SCOPES = CUDA_THREAD_SCOPE_THREAD CUDA_THREAD_SCOPE_DEVICE CUDA_THREAD_SCOPE_BLOCK # CUDA_THREAD_SCOPE_SYSTEM #
# SIZES = uint64_t uint32_t uint16_t uint8_t
SIZES = DATA_SIZE_32 DATA_SIZE_64  DATA_SIZE_16 DATA_SIZE_8
# BUFFER = BUFFER_SAME BUFFER_DIFF

# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(foreach signal,$(SIGNAL_SCOPES),$(OUTPUT)_$(scope)_$(size)_$(buf)_$(signal).out))))

all:  $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_acq_rel_$(scope)_$(size).out)) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_acq_rlx_$(scope)_$(size).out)) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_rlx_rel_$(scope)_$(size).out)) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_rlx_rlx_$(scope)_$(size).out))

output:
	mkdir -p output

acq-rel: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_acq_rel_$(scope)_$(size).out))

acq-rlx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_acq_rlx_$(scope)_$(size).out))

rlx-rel: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_rlx_rel_$(scope)_$(size).out))

rlx-rlx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(OUTPUT)_rlx_rlx_$(scope)_$(size).out))

define make_target
$(OUTPUT)_$(1)_$(2).out: $(SRC) $(HEADERS) | output
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -o $$@ $$(SRC)
endef

define make_target_acq_rel
$(OUTPUT)_acq_rel_$(1)_$(2).out: $(SRC) $(HEADERS) | output
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -DP_H_FLAG_STORE_ORDER_REL -DC_H_FLAG_LOAD_ORDER_ACQ -o $$@ $$(SRC)
endef

define make_target_acq_rlx
$(OUTPUT)_acq_rlx_$(1)_$(2).out: $(SRC) $(HEADERS) | output
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -DP_H_FLAG_STORE_ORDER_RLX -DC_H_FLAG_LOAD_ORDER_ACQ -o $$@ $$(SRC)
endef

define make_target_rlx_rel
$(OUTPUT)_rlx_rel_$(1)_$(2).out: $(SRC) $(HEADERS) | output
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -DP_H_FLAG_STORE_ORDER_REL -DC_H_FLAG_LOAD_ORDER_RLX -o $$@ $$(SRC)
endef

define make_target_rlx_rlx
$(OUTPUT)_rlx_rlx_$(1)_$(2).out: $(SRC) $(HEADERS) | output
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -DP_H_FLAG_STORE_ORDER_RLX -DC_H_FLAG_LOAD_ORDER_RLX -o $$@ $$(SRC)
endef

$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target,$(scope),$(size)))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target_acq_rel,$(scope),$(size)))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target_acq_rlx,$(scope),$(size)))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target_rlx_rel,$(scope),$(size)))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(eval $(call make_target_rlx_rlx,$(scope),$(size)))))

clean:
	rm -r output
	rm -f *.out *.ptx

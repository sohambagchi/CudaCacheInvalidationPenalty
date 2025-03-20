OUTPUT = cache_invalidation_testing
SRC = cache_invalidation_testing.cu
HEADERS = cache_invalidation_testing.cuh 
NVCC = nvcc
NVCC_FLAGS = -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87
LIBS = -lnuma -lm

# SCOPES = cuda\:\:thread_scope_system cuda\:\:thread_scope_thread cuda\:\:thread_scope_device # cuda\:\:thread_scope_block

SCOPES = CUDA_THREAD_SCOPE_THREAD #CUDA_THREAD_SCOPE_DEVICE CUDA_THREAD_SCOPE_BLOCK # CUDA_THREAD_SCOPE_SYSTEM #
# SIZES = uint64_t uint32_t uint16_t uint8_t
SIZES = DATA_SIZE_32 #DATA_SIZE_64  DATA_SIZE_16 DATA_SIZE_8
BUFFER = BUFFER_SAME #BUFFER_DIFF

# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(foreach signal,$(SIGNAL_SCOPES),$(OUTPUT)_$(scope)_$(size)_$(buf)_$(signal).out))))

all:  $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rel_$(scope)_$(size)_$(buf).out))) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rlx_$(scope)_$(size)_$(buf).out))) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rel_$(scope)_$(size)_$(buf).out))) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rlx_$(scope)_$(size)_$(buf).out)))

flag-rel: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rel_$(scope)_$(size)_$(buf).out)))

flag-rlx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rlx_$(scope)_$(size)_$(buf).out)))

no-acq-flag-rel: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rel_$(scope)_$(size)_$(buf).out)))

no-acq-flag-rlx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rlx_$(scope)_$(size)_$(buf).out)))

define make_target
$(OUTPUT)_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -o $$@ $$(SRC)
endef

define make_target_rel
$(OUTPUT)_rel_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DP_H_FLAG_STORE_ORDER_REL  -o $$@ $$(SRC)
endef

define make_target_rlx
$(OUTPUT)_rlx_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DP_H_FLAG_STORE_ORDER_RLX  -o $$@ $$(SRC)
endef

define make_target_no_acq_rel
$(OUTPUT)_no_acq_rel_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DNO_ACQ -DP_H_FLAG_STORE_ORDER_REL  -o $$@ $$(SRC)
endef

define make_target_no_acq_rlx
$(OUTPUT)_no_acq_rlx_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DNO_ACQ -DP_H_FLAG_STORE_ORDER_RLX  -o $$@ $$(SRC)
endef

$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target,$(scope),$(size),$(buf))))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_rel,$(scope),$(size),$(buf))))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_rlx,$(scope),$(size),$(buf))))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_no_acq_rel,$(scope),$(size),$(buf))))))
$(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_no_acq_rlx,$(scope),$(size),$(buf))))))

clean:
	rm -f *.out *.ptx

NVCC := nvcc
NVCCFLAGS := -O2 -std=c++17
NVCCFLAGS += -Iinclude  # header files are in the include directory
NVCCFLAGS += -Isrc # src

SRC := main.cu
FIELD_SRC := bin/field25519_test.cu src/kernels.cu
FIELD_SQRT_SRC := bin/field25519_sqrt_test.cu src/kernels.cu
ALLEGATOR_SRC := bin/allegator25519_compare_test.cu src/kernels.cu
SCALARMULT_SRC := bin/curve25519_scalarmult_test.cu src/kernels.cu
OPRF_SRC := bin/OPRF.cu src/kernels.cu
LDLIBS := -lcrypto
SODIUMLIBS := -L/lib/x86_64-linux-gnu -l:libsodium.so.23

# Where the executables are compiled
TARGET_DIR := bin

TARGET := $(TARGET_DIR)/vector_mul
FIELD_TARGET := $(TARGET_DIR)/field25519_test
FIELD_SQRT_TARGET := $(TARGET_DIR)/field25519_sqrt_test
ALLEGATOR_TARGET := $(TARGET_DIR)/allegator25519_compare_test
SCALARMULT_TARGET := $(TARGET_DIR)/curve25519_scalarmult_test
OPRF_TARGET := $(TARGET_DIR)/OPRF

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

ifdef CUDA_ARCH
NVCCFLAGS += -arch=$(CUDA_ARCH)
endif

all: $(TARGET) $(FIELD_TARGET) $(FIELD_SQRT_TARGET) $(ALLEGATOR_TARGET) $(SCALARMULT_TARGET) $(OPRF_TARGET)

$(TARGET): $(SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC) $(LDLIBS)

run: $(TARGET)
	./$(TARGET) $(ARGS)

$(FIELD_TARGET): $(FIELD_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(FIELD_TARGET) $(FIELD_SRC) $(LDLIBS)

field_test: $(FIELD_TARGET)
	./$(FIELD_TARGET) $(ARGS)

$(FIELD_SQRT_TARGET): $(FIELD_SQRT_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(FIELD_SQRT_TARGET) $(FIELD_SQRT_SRC)

field_sqrt_test: $(FIELD_SQRT_TARGET)
	./$(FIELD_SQRT_TARGET) $(ARGS)

$(ALLEGATOR_TARGET): $(ALLEGATOR_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(ALLEGATOR_TARGET) $(ALLEGATOR_SRC) $(SODIUMLIBS)

allegator_test: $(ALLEGATOR_TARGET)
	./$(ALLEGATOR_TARGET) $(ARGS)

$(SCALARMULT_TARGET): $(SCALARMULT_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(SCALARMULT_TARGET) $(SCALARMULT_SRC) $(SODIUMLIBS)

scalarmult_test: $(SCALARMULT_TARGET)
	./$(SCALARMULT_TARGET) $(ARGS)

$(OPRF_TARGET): $(OPRF_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OPRF_TARGET) $(OPRF_SRC) $(LDLIBS) $(SODIUMLIBS)

oprf: $(OPRF_TARGET)
	./$(OPRF_TARGET) $(ARGS)

clean:
	rm -f $(TARGET) $(FIELD_TARGET) $(FIELD_SQRT_TARGET) $(ALLEGATOR_TARGET) $(SCALARMULT_TARGET) $(OPRF_TARGET)

.PHONY: all run field_test field_sqrt_test allegator_test scalarmult_test oprf clean

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
OPRF_SENDER_SRC := bin/oprf_sender.cu src/kernels.cu
OPRF_RECEIVER_SRC := bin/oprf_receiver.cu src/kernels.cu
LDLIBS := -lcrypto
SODIUMLIBS := -L/lib/x86_64-linux-gnu -l:libsodium.so.23

# Where the executables are compiled
TARGET_DIR := target

TARGET := $(TARGET_DIR)/vector_mul
FIELD_TARGET := $(TARGET_DIR)/field25519_test
FIELD_SQRT_TARGET := $(TARGET_DIR)/field25519_sqrt_test
ALLEGATOR_TARGET := $(TARGET_DIR)/allegator25519_compare_test
SCALARMULT_TARGET := $(TARGET_DIR)/curve25519_scalarmult_test
OPRF_TARGET := $(TARGET_DIR)/OPRF
OPRF_SENDER_TARGET := $(TARGET_DIR)/oprf_sender
OPRF_RECEIVER_TARGET := $(TARGET_DIR)/oprf_receiver

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

ifdef CUDA_ARCH
NVCCFLAGS += -arch=$(CUDA_ARCH)
endif

all: $(TARGET) $(FIELD_TARGET) $(FIELD_SQRT_TARGET) $(ALLEGATOR_TARGET) $(SCALARMULT_TARGET) $(OPRF_TARGET) $(OPRF_SENDER_TARGET) $(OPRF_RECEIVER_TARGET)

$(TARGET): $(SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC) $(LDLIBS)

run: $(TARGET)
	./$(TARGET) $(ARGS)

$(FIELD_TARGET): $(FIELD_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(FIELD_TARGET) $(FIELD_SRC) $(LDLIBS)

field25519_test: $(FIELD_TARGET)

field_test_run: $(FIELD_TARGET)
	./$(FIELD_TARGET) $(ARGS)

$(FIELD_SQRT_TARGET): $(FIELD_SQRT_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(FIELD_SQRT_TARGET) $(FIELD_SQRT_SRC)

field25519_sqrt_test: $(FIELD_SQRT_TARGET)

field_sqrt_test_run: $(FIELD_SQRT_TARGET)
	./$(FIELD_SQRT_TARGET) $(ARGS)

$(ALLEGATOR_TARGET): $(ALLEGATOR_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(ALLEGATOR_TARGET) $(ALLEGATOR_SRC) $(SODIUMLIBS)

allegator25519_compare_test: $(ALLEGATOR_TARGET)

allegator_test_run: $(ALLEGATOR_TARGET)
	./$(ALLEGATOR_TARGET) $(ARGS)

$(SCALARMULT_TARGET): $(SCALARMULT_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(SCALARMULT_TARGET) $(SCALARMULT_SRC) $(SODIUMLIBS)

curve25519_scalarmult_test: $(SCALARMULT_TARGET)

scalarmult_test_run: $(SCALARMULT_TARGET)
	./$(SCALARMULT_TARGET) $(ARGS)

$(OPRF_TARGET): $(OPRF_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OPRF_TARGET) $(OPRF_SRC) $(LDLIBS) $(SODIUMLIBS)

OPRF: $(OPRF_TARGET)

oprf_run: $(OPRF_TARGET)
	./$(OPRF_TARGET) $(ARGS)

$(OPRF_SENDER_TARGET): $(OPRF_SENDER_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OPRF_SENDER_TARGET) $(OPRF_SENDER_SRC) $(LDLIBS)

oprf_sender: $(OPRF_SENDER_TARGET)

oprf_sender_run: $(OPRF_SENDER_TARGET)
	./$(OPRF_SENDER_TARGET) $(ARGS)

$(OPRF_RECEIVER_TARGET): $(OPRF_RECEIVER_SRC) | $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(OPRF_RECEIVER_TARGET) $(OPRF_RECEIVER_SRC) $(LDLIBS)

oprf_receiver: $(OPRF_RECEIVER_TARGET)

oprf_receiver_run: $(OPRF_RECEIVER_TARGET)
	./$(OPRF_RECEIVER_TARGET) $(ARGS)

clean:
	rm -f $(TARGET) $(FIELD_TARGET) $(FIELD_SQRT_TARGET) $(ALLEGATOR_TARGET) $(SCALARMULT_TARGET) $(OPRF_TARGET) $(OPRF_SENDER_TARGET) $(OPRF_RECEIVER_TARGET)

.PHONY: all run field25519_test field25519_sqrt_test allegator25519_compare_test curve25519_scalarmult_test OPRF oprf_sender oprf_receiver field_test_run field_sqrt_test_run allegator_test_run scalarmult_test_run oprf_run oprf_sender_run oprf_receiver_run clean

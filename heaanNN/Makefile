PROJECT_ROOT := $(shell cd .. && pwd)

HEAAN_SRC  := $(PROJECT_ROOT)/src/HEAAN-PRNG-Control

BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin
OBJ_DIR   := $(BUILD_DIR)/obj

CXX := g++

CXXFLAGS := -std=c++17 -O2 -pthread -MMD -MP \
            -I$(HEAAN_SRC)/src \
            -Isrc

LDFLAGS := -L$(HEAAN_SRC)/lib -L/usr/local/lib
LIBS    := -lHEAAN -lntl -lgmp -lm



MAINS := $(filter-out src/utils%.cpp,$(wildcard src/*.cpp))

PROGRAMS := $(notdir $(MAINS:.cpp=))
TARGETS  := $(addprefix $(BIN_DIR)/,$(PROGRAMS))



UTILS_SRC := $(filter src/utils%.cpp,$(wildcard src/*.cpp))

UTILS_OBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(UTILS_SRC))



BIN     := ./build/bin/nn_mnist
COMMON_PARAMS := --logN 12 --logQ 220 --bitPerCoeff 250 --logDelta 30 --logSlots 10 --withNTT 0

.PHONY: all run0


all: dirs $(TARGETS)
# Is like this to paralelize with -j9
run0:
	$(BIN) $(COMMON_PARAMS) --stage encrypt_c0 --seed 0


dirs:
	@mkdir -p $(BIN_DIR)


$(BIN_DIR)/%: src/%.cpp $(UTILS_OBJ)
	@mkdir -p $(dir $@)

	$(CXX) $(CXXFLAGS) -c $< -o $(OBJ_DIR)/$<.o

	$(CXX) \
	    $(OBJ_DIR)/$<.o \
	    $(UTILS_OBJ) \
	    -o $@ \
	    $(LDFLAGS) $(LIBS)

	@echo "✓ Built $@"




$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@



clean:
	rm -rf $(BUILD_DIR)






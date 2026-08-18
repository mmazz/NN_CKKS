PROJECT_ROOT := $(shell cd ../.. && pwd)

HEAAN_SRC  := $(PROJECT_ROOT)/src/HEAAN-PRNG-Control
COMMON_SRC := $(PROJECT_ROOT)/src/common

BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin
OBJ_DIR   := $(BUILD_DIR)/obj

CXX := g++

CXXFLAGS := -std=c++17 -O2 -pthread -MMD -MP \
            -I$(HEAAN_SRC)/src \
            -I$(COMMON_SRC) \
            -Isrc

LDFLAGS := -L$(HEAAN_SRC)/lib -L/usr/local/lib
LIBS    := -lHEAAN -lntl -lgmp -lm


# ------------------------------------------------
# Detect mains automatically
# (todo archivo que NO empiece con utils)
# ------------------------------------------------

MAINS := $(filter-out src/utils%.cpp,$(wildcard src/*.cpp))

PROGRAMS := $(notdir $(MAINS:.cpp=))
TARGETS  := $(addprefix $(BIN_DIR)/,$(PROGRAMS))


# ------------------------------------------------
# Utils / shared code
# ------------------------------------------------

UTILS_SRC := $(filter src/utils%.cpp,$(wildcard src/*.cpp)) \
             $(wildcard $(COMMON_SRC)/*.cpp)

UTILS_OBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(UTILS_SRC))


# ------------------------------------------------

BIN_RANDOM     := ./build/bin/nn_mnist_randomBitFlip
SEEDS := 1 2 3
RUN_IDS_CLIENT := 0 1 2 3 4 5
RUN_IDS_SERVER := 0 1 2 3 4 5 6 7 8 9
RUNS := dirs $(addprefix run,$(RUN_IDS))
RUNS_CLIENT := dirs $(addprefix run_stages,$(RUN_IDS_CLIENT))
RUNS_SERVER := dirs $(addprefix run_server,$(RUN_IDS_SERVER))
RUNS_SERVER_LAYERS:= dirs $(addprefix run_server_layer,$(RUN_IDS))
RUN_IDS_new := 1 2
RUNS_SERVER_LAYERS_new := dirs $(addprefix run_server_layer_new,$(RUN_IDS_new))

COMMON_PARAMS := --logN 12 --logQ 220 --bitPerCoeff 250 --logDelta 30 --logSlots 10 --withNTT 0

.PHONY: all allRuns $(RUNS) allRuns_client $(RUNS_CLIENT) allRuns_server $(RUNS_SERVER)  allRuns_server_layers $(RUNS_SERVER_LAYERS)


all: dirs $(TARGETS)

allRuns_server_layers_new: $(RUNS_SERVER_LAYERS_new)

run_server_layer_new1:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 2 ; \
	done

run_server_layer_new2:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 3 ; \
	done
allRuns: $(RUNS)
allRuns_client: $(RUNS_CLIENT)
allRuns_server: $(RUNS_SERVER)
allRuns_server_layers: $(RUNS_SERVER_LAYERS)

# Is like this to paralelize with -j9
run0:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 0
run1:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 1
run2:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 2
run3:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 3
run4:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 4
run5:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 5
run6:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 6
run7:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 7
run8:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 8
run9:
	$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed 9

##
## CLIENT RUNS
##

run_stages0:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
			$(BIN_RANDOM) $(COMMON_PARAMS) --stage encode  --seed $$s --seed_input $$s_input; \
		done; \
	done

run_stages1:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
			$(BIN_RANDOM) $(COMMON_PARAMS) --stage encrypt_c0 --seed $$s --seed_input $$s_input ; \
		done; \
	done

run_stages2:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM)  $(COMMON_PARAMS) --stage encrypt_c1 --seed $$s --seed_input $$s_input ; \
		done; \
	done

run_stages3:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage decrypt_c0 --seed $$s --seed_input $$s_input; \
		done; \
	done

run_stages4:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage decrypt_c1 --seed $$s --seed_input $$s_input; \
		done; \
	done

run_stages5:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage decode --seed $$s --seed_input $$s_input; \
		done; \
	done
##
## SERVER RUNS
##
run_server0:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 0 ; \
		done; \
	done

run_server1:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 1 ; \
		done; \
	done
run_server2:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input  --op_step 2 ; \
		done; \
	done

run_server3:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 3 ; \
		done; \
	done

run_server4:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input  --op_step 4 ; \
		done; \
	done

run_server5:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 5 ; \
		done; \
	done
run_server6:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input  --op_step 6 ; \
		done; \
	done

run_server7:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 7 ; \
		done; \
	done

run_server8:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 8 ; \
		done; \
	done

run_server9:
	for s in $(SEEDS); do \
		for s_input in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage cheby_tanh3 --seed $$s --seed_input $$s_input --op_step 9 ; \
		done; \
	done


## HIDDEN LAYER
run_server_layer0:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 0 ; \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 1 ; \
	done
run_server_layer1:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 2 ; \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 3 ; \
	done
run_server_layer2:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 4 ; \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 5 ; \
	done
run_server_layer3:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 6 ; \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 7 ; \
	done
run_server_layer4:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 8 ; \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 9 ; \
	done
run_server_layer5:
	for s in $(SEEDS); do \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 10 ; \
		$(BIN_RANDOM) $(COMMON_PARAMS) --stage hidden_layer --seed $$s --seed_input $$s --op_step 11 ; \
	done
dirs:
	@mkdir -p $(BIN_DIR)


# ------------------------------------------------
# Link rule PER TARGET
# solo linkea SU main + utils
# ------------------------------------------------

$(BIN_DIR)/%: src/%.cpp $(UTILS_OBJ)
	@mkdir -p $(dir $@)

	$(CXX) $(CXXFLAGS) -c $< -o $(OBJ_DIR)/$<.o

	$(CXX) \
	    $(OBJ_DIR)/$<.o \
	    $(UTILS_OBJ) \
	    -o $@ \
	    $(LDFLAGS) $(LIBS)

	@echo "✓ Built $@"


# ------------------------------------------------
# Compile utils
# ------------------------------------------------

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@



clean:
	rm -rf $(BUILD_DIR)


run0OP:
	build/bin/nn_mnist_randomBitFlipOp --logN 15  --logQ 336  --bitPerCoeff 250  --logDelta 30 --stage encrypt_c0 --logSlots 10 --seed 2 --withNTT 0 --doMul 1 --verbose

run:
	build/bin/nn_mnist 6 0 >> log.txt
	build/bin/nn_mnist 7 0 >> log.txt
	build/bin/nn_mnist 8 0 >> log.txt
	build/bin/nn_mnist 9 0 >> log.txt
	build/bin/nn_mnist 10 0 >> log.txt
	build/bin/nn_mnist 11 0 >> log.txt



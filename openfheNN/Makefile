PROJECT_ROOT := $(shell cd ../.. && pwd)
RUN_IDS := 0 1 2 3
RUNS := dirs $(addprefix run,$(RUN_IDS))
RUNS_nonNTT := dirs $(addprefix run_nonNTT,$(RUN_IDS))

.PHONY: allRuns $(RUNS)
BIN     := ./build/bin/nn_mnist
SEEDS := 1 2 3 4 5
RUNS := dirs $(addprefix run,$(RUN_IDS))
RUNS_SEEDS := dirs $(addprefix run_stages,$(RUN_IDS))

.PHONY: all allRuns $(RUNS)

all: dirs $(TARGETS)

allRuns: $(RUNS)
allRuns_seeds: $(RUNS_SEEDS)



allRuns: $(RUNS)


run0:
	$(BIN) --logN 12  --logQ 60  --bitPerCoeff 64  --logDelta 50   --logSlots 10 --seed 0 --withNTT 1 --mult_depth 5

dirs:
	@mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR)





#pragma once
#include "HEAAN.h"
#include <NTL/ZZ.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include "campaign_helper.h"
#include "backend_interface.h"
#include "utils_ckks.h"

struct HEEnv {
    Context context;
    SecretKey sk;
    Scheme scheme;
    vector<long> rotIdx;

    HEEnv(long logN, long logQ, long h)
        : context(logN, logQ),
          sk(logN, h),
          scheme(sk, context)
    {
        for (long p = 1; p <= 512; p <<= 1) {
            rotIdx.push_back(p);
            scheme.addLeftRotKey(sk, p);
        }
    }
};

struct EncodedWeights {
    vector<ZZX> W1;
    vector<double> b1;

    vector<vector<ZZX>> W2;
    vector<double> b2;
};


EncodedWeights encodeWeights(
    HEEnv& he,
    const vector<vector<double>>& W1,
    const vector<double>& b1,
    const vector<vector<double>>& W2,
    const vector<double>& b2,
    long slots,
    long logP
);

Ciphertext encryptInput(
    HEEnv& he,
    const vector<double>& vals,
    long slots,
    long logP,
    long logQ
);

void reduceSum(
    HEEnv& he,
    Ciphertext& ct,
    long logSlots,
    uint32_t &reduceSum_layer,
    CampaignArgs& args, std::optional<IterationArgs> iterArgs
);

vector<Plaintext> decryptLogits(
    HEEnv& he,
    const vector<Ciphertext>& outs
);

vector<double> decodeLogits(
    HEEnv& he,
    vector<Plaintext>& outs
);


bool loadMnistNormRowByIndex(const std::string &csvPath, size_t rowIndex,
                         size_t &outLabel, std::vector<double> &pixelsOut);


std::vector<std::vector<double>> loadCSVMatrix(const std::string& path, size_t rows, size_t cols);

std::vector<double> loadCSVVector(const std::string& path, size_t size);

IterationResult run_iteration_NN(HEEnv& he, EncodedWeights encoded,
        const vector<double>& vals, CampaignArgs& args, size_t targetValue,
        uint32_t &hidden_layer, uint32_t &reduceSum_layer,
        std::optional<IterationArgs> iterArgs=std::nullopt
);

Ciphertext chebyTanh3(
    HEEnv& he,
    Ciphertext c,
    long logP, CampaignArgs& args, uint32_t hidden,
     std::optional<IterationArgs> iterArgs
);

IterationResult run_iteration_NNOp(HEEnv& he, EncodedWeights encoded,
        const vector<double>& vals, CampaignArgs& args, size_t targetValue,
        std::optional<IterationArgs> iterArgs=std::nullopt
        );

vector<Ciphertext> forward(
    HEEnv& he,
    Ciphertext c,
    EncodedWeights& ew,
    long logSlots,
    long logP,
    uint32_t &hidden_layer,
    uint32_t &reduceSum_layer,
    CampaignArgs& args, std::optional<IterationArgs> iterArgs
);

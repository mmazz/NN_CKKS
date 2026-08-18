#pragma once
#include "HEAAN.h"
#include <NTL/ZZ.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <getopt.h>
#include <cassert>

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


struct CampaignArgs {
    std::string library = "none";
    std::string stage = "none";

    uint32_t bitPerCoeff = 64;
    uint32_t logN = 3;
    uint32_t logQ = 60;
    uint32_t logDelta = 50;
    uint32_t logSlots = 2;
    uint32_t mult_depth = 0;
    uint32_t logMin = 0;
    uint32_t logMax = 0;

    uint32_t seed = 0;
    uint32_t seed_input = 0;

    bool withNTT = false;
    uint32_t doAdd = false;
    uint32_t doPlainMul = 0;
    uint32_t doMul = 0;
    double doScalarMul = 0;
    uint32_t doRot = 0;
    uint32_t doBoot = 0;
    uint32_t op_step = 0;
    uint32_t op_depth = 0;
    size_t isComplex = 0;
    bool verbose = false;
    uint32_t dnum = 3;
    std::string scaleTech = "FIXEDMANUAL";
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
    CampaignArgs& args
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

CampaignArgs parse_arguments(int argc, char* argv[]);
int run_iteration_NN(HEEnv& he, EncodedWeights encoded,
        const vector<double>& vals, CampaignArgs& args, size_t targetValue
);

Ciphertext chebyTanh3(
    HEEnv& he,
    Ciphertext c,
    long logP, CampaignArgs& args
);

vector<Ciphertext> forward(
    HEEnv& he,
    Ciphertext c,
    EncodedWeights& ew,
    long logSlots,
    long logP,
    CampaignArgs& args
);

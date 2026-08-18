#pragma once
#include "lattice/hal/lat-backend.h"
#include "math/hal/nativeintbackend.h"
#include "openfhe.h"
#include <NTL/ZZ.h>
#include <vector>
#include <algorithm>

#include <getopt.h>
#include <cassert>
using namespace lbcrypto;
using namespace std;

struct HEEnv {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
    std::vector<int> rotIdx;

    HEEnv(uint32_t logN,
          uint32_t multDepth,
          uint32_t scaleMod,
          uint32_t firstMod) {
        auto cfg = SDCConfigHelper::MakeConfig(
            false, // Disable execption
            SecretKeyAttackMode::CompleteInjection
        );

        SDCConfigHelper::SetGlobalConfig(cfg);
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(multDepth);
        parameters.SetScalingModSize(scaleMod);
        parameters.SetFirstModSize(firstMod);
        parameters.SetBatchSize(1 << (logN-1));
        parameters.SetRingDim(1 << logN);
        //parameters.SetScalingTechnique(FIXEDMANUAL);
        parameters.SetSecurityLevel(HEStd_NotSet);

        cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(LEVELEDSHE);

        keys = cc->KeyGen();

        cc->EvalMultKeyGen(keys.secretKey);

        for (int p = 1; p <= 512; p <<= 1) {
            rotIdx.push_back(p);
        }
        cc->EvalAtIndexKeyGen(keys.secretKey, rotIdx);
    }
};

struct EncodedWeights {
    std::vector<Plaintext> W1;
    std::vector<Plaintext> b1;

    std::vector<std::vector<Plaintext>> W2;
    std::vector<Plaintext> b2;
};
EncodedWeights encodeWeights(
    HEEnv& he,
    const vector<vector<double>>& W1,
    const vector<double>& b1,
    const vector<vector<double>>& W2,
    const vector<double>& b2,
    size_t slots
);

Ciphertext<DCRTPoly> encryptInput(
    HEEnv& he,
    const std::vector<double>& vals
) ;


Ciphertext<DCRTPoly> chebyTanh3(
    HEEnv& he,
    Ciphertext<DCRTPoly> x
);

Ciphertext<DCRTPoly> reduceSum(
    HEEnv& he,
    Ciphertext<DCRTPoly> ct
);


vector<Ciphertext<DCRTPoly>> forward(
    HEEnv& he,
    Ciphertext<DCRTPoly> x,
    EncodedWeights& ew,
    long logSlots,
    long logP
);

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


CampaignArgs parse_arguments(int argc, char* argv[]);

int run_iteration_NN(
    HEEnv& he,
    EncodedWeights encoded,
    const vector<double>& vals,
    CampaignArgs& args,
    size_t targetValue
) ;


vector<double> decryptLogits(HEEnv& he, vector<Ciphertext<DCRTPoly>>& outs
);

bool loadMnistNormRowByIndex(const std::string &csvPath, size_t rowIndex,
                         size_t &outLabel, std::vector<double> &pixelsOut);


std::vector<std::vector<double>> loadCSVMatrix(const std::string& path, size_t rows, size_t cols);

std::vector<double> loadCSVVector(const std::string& path, size_t size);




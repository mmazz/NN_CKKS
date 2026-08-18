#include "utils_nn.h"
#include "backend_interface.h"

static void bitFlip(Ciphertext<DCRTPoly> &c, bool withNTT, size_t k, size_t i, size_t j, size_t bit){
    if(!withNTT)
        c->GetElements()[k].SwitchFormat();

    NativeInteger& x = c->GetElements()[k].GetAllElements()[i][j];
    uint64_t val = x.ConvertToInt();  // Extrae como uint64_t
    val ^= (1ULL << bit);               // Aplica XOR
    x = NativeInteger(val);

    if(!withNTT)
        c->GetElements()[k].SwitchFormat();
}

static void bitFlip(Plaintext &ptxt, bool withNTT, size_t i, size_t j, size_t bit){
    if(!withNTT)
        ptxt->GetElement<DCRTPoly>().SwitchFormat();

    NativeInteger& x = ptxt->GetElement<DCRTPoly>().GetAllElements()[i][j];
    uint64_t val = x.ConvertToInt();  // Extrae como uint64_t
    val ^= (1ULL << bit);               // Aplica XOR
    x = NativeInteger(val);

    if(!withNTT)
        ptxt->GetElement<DCRTPoly>().SwitchFormat();
}

EncodedWeights encodeWeights(
    HEEnv& he,
    const vector<vector<double>>& W1,
    const vector<double>& b1,
    const vector<vector<double>>& W2,
    const vector<double>& b2,
    size_t slots
){
    EncodedWeights ew;

    vector<double> buffer(slots);

    // W1
    for (size_t j = 0; j < W1.size(); ++j) {
        fill(buffer.begin(), buffer.end(), 0.0);
        for (size_t i = 0; i < W1[j].size(); ++i)
            buffer[i] = W1[j][i];

        ew.W1.push_back(he.cc->MakeCKKSPackedPlaintext(buffer));
    }

    // b1
    for (auto v : b1) {
        fill(buffer.begin(), buffer.end(), v);
        ew.b1.push_back(he.cc->MakeCKKSPackedPlaintext(buffer));
    }

    // W2
    ew.W2.resize(W2.size());
    for (size_t o = 0; o < W2.size(); ++o) {
        ew.W2[o].resize(W2[o].size());
        for (size_t h = 0; h < W2[o].size(); ++h) {
            fill(buffer.begin(), buffer.end(), W2[o][h]);
            ew.W2[o][h] = he.cc->MakeCKKSPackedPlaintext(buffer);
        }
    }

    // b2
    for (auto v : b2) {
        fill(buffer.begin(), buffer.end(), v);
        ew.b2.push_back(he.cc->MakeCKKSPackedPlaintext(buffer));
    }

    return ew;
}
Ciphertext<DCRTPoly> encryptInput(
    HEEnv& he,
    const vector<double>& vals,
    long slots,
    long logP,
    long logQ
){
    vector<complex<double>> arr(slots, {0,0});

    for(size_t i=0;i<vals.size();++i)
        arr[i] = {vals[i],0};

    Plaintext pt = he.cc->MakeCKKSPackedPlaintext(arr);

    return he.cc->Encrypt(he.keys.publicKey, pt);
}

Ciphertext<DCRTPoly> chebyTanh3(
    HEEnv& he,
    Ciphertext<DCRTPoly> x
) {
    auto x2 = he.cc->EvalMult(x, x);
    auto x3 = he.cc->EvalMult(x2, x);

    auto t1 = he.cc->EvalMult(x3, -0.23);
    auto t2 = he.cc->EvalMult(x, 0.98);

    return he.cc->EvalAdd(t1, t2);
}



Ciphertext<DCRTPoly> reduceSum(
    HEEnv& he,
    Ciphertext<DCRTPoly> ct,
    size_t logSlots
) {
    for (size_t i = 0; i < logSlots; ++i) {
        auto rot = he.cc->EvalRotate(ct, 1 << i);
        ct = he.cc->EvalAdd(ct, rot);
    }
    return ct;
}

vector<Ciphertext<DCRTPoly>> forward(
    HEEnv& he,
    Ciphertext<DCRTPoly> c,
    EncodedWeights& ew,
    size_t logSlots
)
{
    size_t HIDDEN = ew.W1.size();
    size_t OUTPUT = ew.W2.size();

    vector<Ciphertext<DCRTPoly>> layer1(HIDDEN);

    // ===== Layer 1 =====
    for (size_t j = 0; j < HIDDEN; ++j) {

        // multByPoly
        auto s = he.cc->EvalMult(c, ew.W1[j]);

        // IMPORTANTE: rescale manual
        he.cc->RescaleInPlace(s);

        // reduceSum SIMD
        for (size_t i = 0; i < logSlots; ++i) {
            auto rot = he.cc->EvalRotate(s, 1 << i);
            s = he.cc->EvalAdd(s, rot);
        }

        // bias
        s = he.cc->EvalAdd(s, ew.b1[j]);

        // Chebyshev
        s = chebyTanh3(he, s);

        layer1[j] = s;
    }

    // ===== Layer 2 =====
    vector<Ciphertext<DCRTPoly>> out(OUTPUT);

    for (size_t o = 0; o < OUTPUT; ++o) {

        auto acc = he.cc->EvalMult(layer1[0], ew.W2[o][0]);
        he.cc->RescaleInPlace(acc);

        for (size_t h = 1; h < HIDDEN; ++h) {

            auto term = he.cc->EvalMult(layer1[h], ew.W2[o][h]);
            he.cc->RescaleInPlace(term);

            acc = he.cc->EvalAdd(acc, term);
        }

        acc = he.cc->EvalAdd(acc, ew.b2[o]);

        out[o] = acc;
    }

    return out;
}
vector<double> decryptLogits(
    HEEnv& he,
    vector<Ciphertext<DCRTPoly>>& outs
){
    vector<double> res(outs.size());

    size_t batchSize =
        he.cc->GetEncodingParams()->GetBatchSize();

    for(size_t i = 0; i < outs.size(); ++i){

        Plaintext dec;
        he.cc->Decrypt(he.keys.secretKey, outs[i], &dec);

        // importante en CKKS
        dec->SetLength(batchSize);

        auto vals = dec->GetCKKSPackedValue();

        res[i] = vals[0].real();
    }

    return res;
}

bool loadMnistNormRowByIndex(const std::string &csvPath, size_t rowIndex,
                         size_t &outLabel, std::vector<double> &pixelsOut)
{
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: no se pudo abrir " << csvPath << "\n";
        return false;
    }

    std::string line;
    size_t currentRow = 0;

    while (std::getline(file, line)) {
        if (currentRow == rowIndex) {

            std::stringstream ss(line);
            std::string cell;

            // label
            if (!std::getline(ss, cell, ',')) {
                std::cerr << "Error: fila vacía en índice " << rowIndex << "\n";
                return false;
            }

            outLabel = std::stoi(cell);

            pixelsOut.clear();
            pixelsOut.reserve(784);

            // -------- normalization parameters --------
            constexpr double inv255 = 1.0 / 255.0;

            // Map to [-1, 1]  (BEST for Chebyshev)
            // x_norm = 2*(x/255) - 1

            while (std::getline(ss, cell, ',')) {

                int pixel = std::stoi(cell);
                pixel = std::clamp(pixel, 0, 255);

                double x = static_cast<double>(pixel) * inv255; // [0,1]
                x = 2.0 * x - 1.0;                              // [-1,1]

                pixelsOut.push_back(x);
            }

            if (pixelsOut.size() != 784) {
                std::cerr << "Error: fila " << rowIndex
                          << " tiene " << pixelsOut.size()
                          << " píxeles (esperado: 784)\n";
                return false;
            }

            return true;
        }

        ++currentRow;
    }

    std::cerr << "Error: índice " << rowIndex
              << " fuera de rango (total filas: " << currentRow << ")\n";

    return false;
}



std::vector<std::vector<double>> loadCSVMatrix(const std::string& path, size_t rows, size_t cols) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir " + path);
    }

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::string line, cell;
    size_t r = 0;

    while (std::getline(file, line) && r < rows) {
        std::stringstream ss(line);
        size_t c = 0;
        while (std::getline(ss, cell, ',') && c < cols) {
            matrix[r][c] = std::stod(cell);
            c++;
        }
        r++;
    }

    return matrix;
}

std::vector<double> loadCSVVector(const std::string& path, size_t size) {
    std::ifstream file(path);
    std::vector<double> data;
    data.reserve(size); // Reservamos para eficiencia

    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir " + path);
    }

    std::string line, cell;
    while (std::getline(file, line) && data.size() < size) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',') && data.size() < size) {
            data.push_back(std::stod(cell));
        }
    }

    if (data.size() != size) {
        throw std::runtime_error("Error: cantidad de valores leídos (" +
                                 std::to_string(data.size()) +
                                 ") no coincide con size esperado (" +
                                 std::to_string(size) + ")");
    }

    return data;
}
IterationResult run_iteration_NN(
    HEEnv& he,
    EncodedWeights encoded,
    const vector<double>& vals,
    CampaignArgs& args,
    size_t targetValue,
    std::optional<IterationArgs> iterArgs
) {
    size_t verbose = args.verbose;

    // ===== Encoding =====
    Plaintext ptxt = he.cc->MakeCKKSPackedPlaintext(vals);

    if (iterArgs && args.stage == "encode") {
        bitFlip(ptxt, args.withNTT,
                iterArgs->limb,
                iterArgs->coeff,
                iterArgs->bit);
    }
    // ===== Encrypt =====
 //   lbcrypto::PseudoRandomNumberGenerator::SetPRNGSeed(args.seed);
    auto c = he.cc->Encrypt(he.keys.publicKey, ptxt);

    if (iterArgs) {
        if (args.stage == "encrypt_c0") {
            bitFlip(c, args.withNTT, 0,
                    iterArgs->limb,
                    iterArgs->coeff,
                    iterArgs->bit);
        } else if (args.stage == "encrypt_c1") {
            bitFlip(c, args.withNTT, 1,
                    iterArgs->limb,
                    iterArgs->coeff,
                    iterArgs->bit);
        }
    }



    if (verbose)
        cout << "Running encrypted inference..." << endl;

    auto outputs = forward(he, c, encoded, args.logSlots);

    if (verbose)
        cout << "Decrypting..." << endl;
    if (iterArgs) {
        if (args.stage == "decrypt_c0") {
            bitFlip(outputs[targetValue], args.withNTT, 0,
                    iterArgs->limb,
                    iterArgs->coeff,
                    iterArgs->bit);
        } else if (args.stage == "decrypt_c1") {
            bitFlip(outputs[targetValue], args.withNTT, 1,
                    iterArgs->limb,
                    iterArgs->coeff,
                    iterArgs->bit);
        }
    }
    auto logits = decryptLogits(he, outputs);

    // ===== Prediction =====
    size_t pred = 0;
    double best = logits[0];

    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best) {
            best = logits[i];
            pred = i;
        }
    }

    IterationResult res;
    res.detected = (pred == targetValue);

    if (verbose) {
        cout << "\nPrediction: " << pred
             << "\nTarget:     " << targetValue << endl;

        cout << (pred == targetValue ? "✔ Correct\n"
                                    : "✘ Incorrect\n");
    } else {
        cout << (pred == targetValue ? 1 : 0) << endl;
    }

    return res;
}



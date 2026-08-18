#include "utils_nn.h"
#include "backend_interface.h"

EncodedWeights encodeWeights(
    HEEnv& he,
    const vector<vector<double>>& W1,
    const vector<double>& b1,
    const vector<vector<double>>& W2,
    const vector<double>& b2,
    long slots,
    long logP
){
    EncodedWeights ew;

    size_t HIDDEN = W1.size();
    size_t INPUT  = W1[0].size();
    size_t OUTPUT = W2.size();

    ew.W1.resize(HIDDEN);

    vector<double> buffer(slots, 0.0);

    for(size_t j=0;j<HIDDEN;++j){
        fill(buffer.begin(), buffer.end(), 0.0);

        for(size_t i=0;i<INPUT;++i)
            buffer[i] = W1[j][i];

        ew.W1[j] = he.context.encode(buffer.data(), slots, logP);
    }

    ew.b1 = b1;

    ew.W2.resize(OUTPUT, vector<ZZX>(HIDDEN));

    for(size_t o=0;o<OUTPUT;++o){
        for(size_t h=0;h<HIDDEN;++h){

            fill(buffer.begin(), buffer.end(), W2[o][h]);

            ew.W2[o][h] = he.context.encode(buffer.data(), slots, logP);
        }
    }

    ew.b2 = b2;

    return ew;
}
Ciphertext encryptInput(
    HEEnv& he,
    const vector<double>& vals,
    long slots,
    long logP,
    long logQ
){
    vector<complex<double>> arr(slots, {0,0});

    for(size_t i=0;i<vals.size();++i)
        arr[i] = {vals[i],0};

    Plaintext pt = he.scheme.encode(arr.data(), slots, logP, logQ);

    return he.scheme.encryptMsg(pt);
}



Ciphertext chebyTanh3(
    HEEnv& he,
    Ciphertext c,
    long logP,
    bool doBitFlip,
    CampaignArgs& args, std::optional<IterationArgs> iterArgs
){
    // x^2

    if (doBitFlip && iterArgs && args.stage == "cheby_tanh3") {
        if (args.op_step== 0) {
            SwitchBit(c.bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.op_step == 1) {
            SwitchBit(c.ax[iterArgs->coeff], iterArgs->bit);
        }
    }
    Ciphertext c2 = he.scheme.square(c);
    he.scheme.reScaleByAndEqual(c2, logP);

    if (doBitFlip && iterArgs && args.stage == "cheby_tanh3") {
        if (args.op_step == 2) {
            SwitchBit(c.bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.op_step == 3) {
            SwitchBit(c.ax[iterArgs->coeff], iterArgs->bit);
        }
    }
    Ciphertext c3 = he.scheme.mult(c2, c);
    he.scheme.reScaleByAndEqual(c3, logP);

    if (doBitFlip && iterArgs && args.stage == "cheby_tanh3") {
        if (args.op_step == 4) {
            SwitchBit(c.bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.op_step == 5) {
            SwitchBit(c.ax[iterArgs->coeff], iterArgs->bit);
        }
    }
    he.scheme.multByConstAndEqual(c3, -0.23, logP);
    he.scheme.reScaleByAndEqual(c3, logP);

    if (doBitFlip && iterArgs && args.stage == "cheby_tanh3") {
        if (args.op_step == 6) {
            SwitchBit(c.bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.op_step == 7) {
            SwitchBit(c.ax[iterArgs->coeff], iterArgs->bit);
        }
    }
    he.scheme.multByConstAndEqual(c, 0.98, logP);
    he.scheme.reScaleByAndEqual(c, logP);

    if (doBitFlip && iterArgs && args.stage == "cheby_tanh3") {
        if (args.op_step == 8) {
            SwitchBit(c.bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.op_step == 9) {
            SwitchBit(c.ax[iterArgs->coeff], iterArgs->bit);
        }
    }
    he.scheme.addAndEqual(c3, c);

    return c3;
}

void reduceSum(
    HEEnv& he,
    Ciphertext& ct,
    long logSlots,
    CampaignArgs& args,
    uint32_t &reduceSum_layer, std::optional<IterationArgs> iterArgs
){
    reduceSum_layer = random_int(0, logSlots-1);

    for(int i=0;i<logSlots;i++){
        Ciphertext rot;
        if (i==reduceSum_layer && iterArgs && args.stage == "hidden_layer") {
            Ciphertext c_copy = ct;
            if (args.op_step == 4) {
                SwitchBit(c_copy.bx[iterArgs->coeff], iterArgs->bit);
            } else if (args.op_step == 5) {
                SwitchBit(c_copy.ax[iterArgs->coeff], iterArgs->bit);
            }
            rot = he.scheme.leftRotateFast(c_copy, 1<<i);
        } else
            rot = he.scheme.leftRotateFast(ct, 1<<i);
        if (i==reduceSum_layer && iterArgs && args.stage == "hidden_layer") {
            if (args.op_step == 6) {
                SwitchBit(rot.bx[iterArgs->coeff], iterArgs->bit);
            } else if (args.op_step == 7) {
                SwitchBit(rot.ax[iterArgs->coeff], iterArgs->bit);
            } else if (args.op_step == 8) {
                SwitchBit(ct.bx[iterArgs->coeff], iterArgs->bit);
            }else if (args.op_step == 9) {
                SwitchBit(ct.ax[iterArgs->coeff], iterArgs->bit);
            }
        }
        he.scheme.addAndEqual(ct, rot);
        if (i==reduceSum_layer && iterArgs && args.stage == "hidden_layer") {
            if (args.op_step == 10) {
                SwitchBit(ct.bx[iterArgs->coeff], iterArgs->bit);
            }else if (args.op_step == 11) {
                SwitchBit(ct.ax[iterArgs->coeff], iterArgs->bit);
            }
        }
    }
}

vector<Ciphertext> forward(
    HEEnv& he,
    Ciphertext c,
    EncodedWeights& ew,
    long logSlots,
    long logP,
    uint32_t &hidden_layer,
    uint32_t &reduceSum_layer,
    CampaignArgs& args, std::optional<IterationArgs> iterArgs
)
{
    size_t HIDDEN = ew.W1.size();
    size_t OUTPUT = ew.W2.size();

    vector<Ciphertext> layer1(HIDDEN);
    hidden_layer = random_int(0, HIDDEN-1);
    for(size_t j=0;j<HIDDEN;++j){
        Ciphertext s;
        if (j==hidden_layer && iterArgs && args.stage == "hidden_layer") {
            Ciphertext c_copy = c;
            if (args.op_step == 0) {
                SwitchBit(c_copy.bx[iterArgs->coeff], iterArgs->bit);
            } else if (args.op_step == 1) {
                SwitchBit(c_copy.ax[iterArgs->coeff], iterArgs->bit);
            }
            s = he.scheme.multByPoly(c_copy, ew.W1[j], logP);
        } else
            s = he.scheme.multByPoly(c, ew.W1[j], logP);

        if (j==hidden_layer && iterArgs && args.stage == "hidden_layer") {
            if (args.op_step == 2) {
                SwitchBit(s.bx[iterArgs->coeff], iterArgs->bit);
            }else if (args.op_step == 3) {
                SwitchBit(s.ax[iterArgs->coeff], iterArgs->bit);
            }
        }

        he.scheme.reScaleByAndEqual(s, logP);

        reduceSum(he, s, logSlots, args, reduceSum_layer, iterArgs);

        he.scheme.addConstAndEqual(s, ew.b1[j]);
        if (j==hidden_layer && iterArgs && args.stage == "hidden_layer") {
            if (args.op_step == 12) {
                SwitchBit(s.bx[iterArgs->coeff], iterArgs->bit);
            }else if (args.op_step == 13) {
                SwitchBit(s.ax[iterArgs->coeff], iterArgs->bit);
            }
        }
        s = chebyTanh3(he, std::move(s), logP, hidden_layer==j, args, iterArgs);

        layer1[j] = std::move(s);
    }

    vector<Ciphertext> out(OUTPUT);
    for(size_t o=0;o<OUTPUT;++o){

        Ciphertext acc = he.scheme.multByPoly(layer1[0], ew.W2[o][0], logP);
        he.scheme.reScaleByAndEqual(acc, logP);

        for(size_t h=0;h<HIDDEN;++h){
            Ciphertext term = he.scheme.multByPoly(layer1[h], ew.W2[o][h], logP);
            he.scheme.reScaleByAndEqual(term, logP);
            he.scheme.addAndEqual(acc, term);
        }

        he.scheme.addConstAndEqual(acc, ew.b2[o]);

        out[o] = std::move(acc);
    }

    return out;
}

vector<Plaintext> decryptLogits(
    HEEnv& he,
    const vector<Ciphertext>& outs
){
    vector<Plaintext> res;
    res.reserve(outs.size());

    for(const auto& ct_const : outs){

        Ciphertext ct = ct_const;  // copia porque decryptMsg no es const
        res.push_back(
            he.scheme.decryptMsg(he.sk, ct)
        );
    }

    return res;
}

vector<double> decodeLogits(
    HEEnv& he,
    vector<Plaintext>& pts
){
    vector<double> res;
    res.reserve(pts.size());

    for(const auto& pt_const : pts){

        Plaintext pt = pt_const;  // copia porque decryptMsg no es const
        unique_ptr<complex<double>[]> tmp(
            he.scheme.decode(pt)
        );

        res.push_back(tmp[0].real());
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


IterationResult run_iteration_NN(HEEnv& he, EncodedWeights encoded,
        const vector<double>& vals, CampaignArgs& args, size_t targetValue,
        uint32_t &hidden_layer,  uint32_t &reduceSum_layer,
        std::optional<IterationArgs> iterArgs ){

    size_t logSlots = args.logSlots;
    size_t slots = 1 << logSlots;
    size_t logP = args.logDelta;
    size_t logQ = args.logQ;
    size_t verbose = args.verbose;


    vector<complex<double>> arr(slots, {0,0});

    for(size_t i=0;i<vals.size();++i)
        arr[i] = {vals[i],0};

    Plaintext plain = he.scheme.encode(arr.data(), slots, logP, logQ);

    if (iterArgs && args.stage == "encode") {
        SwitchBit(plain.mx[iterArgs->coeff], iterArgs->bit);
    }
    Ciphertext c = he.scheme.encryptMsg(plain, ZZ(args.seed));

    if (iterArgs) {
        if (args.stage == "encrypt_c0") {
            SwitchBit(c.bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.stage == "encrypt_c1") {
            SwitchBit(c.ax[iterArgs->coeff], iterArgs->bit);
        }
    }

    if(args.verbose)
        cout << "Running encrypted inference..." << endl;
    auto outputs = forward(
        he,
        c,
        encoded,
        logSlots,
        logP,
        hidden_layer, reduceSum_layer,
        args, iterArgs
    );

    if(verbose)
        cout << "Decrypting..." << endl;

    // I make a bit flip on the cipher with the target value
    if (iterArgs) {
        if (args.stage == "decrypt_c0") {
            SwitchBit(outputs[targetValue].bx[iterArgs->coeff], iterArgs->bit);
        } else if (args.stage == "decrypt_c1") {
            SwitchBit(outputs[targetValue].ax[iterArgs->coeff], iterArgs->bit);
        }
    }
    auto logitsDec = decryptLogits(he, outputs);

    if (iterArgs && args.stage == "decode") {
        SwitchBit(logitsDec[targetValue].mx[iterArgs->coeff], iterArgs->bit);
    }

    auto logits = decodeLogits(he, logitsDec);

    size_t pred = 0;
    double best = logits[0];

    for(size_t i=1;i<logits.size();++i){
        if(logits[i] > best){
            best = logits[i];
            pred = i;
        }
    }
    IterationResult res;
    res.detected = (pred == targetValue);
    if(verbose){
        cout << "\nPrediction: " << pred
             << "\nTarget:     " << targetValue
             << endl;

        if(pred == targetValue)
            cout << "✔ Correct\n";
        else
            cout << "✘ Incorrect\n";
    }
    return res;
}

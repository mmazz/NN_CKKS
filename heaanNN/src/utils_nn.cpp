#include "utils_nn.h"

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
    CampaignArgs& args
){

    Ciphertext c2 = he.scheme.square(c);
    he.scheme.reScaleByAndEqual(c2, logP);

    Ciphertext c3 = he.scheme.mult(c2, c);
    he.scheme.reScaleByAndEqual(c3, logP);

    he.scheme.multByConstAndEqual(c3, -0.23, logP);
    he.scheme.reScaleByAndEqual(c3, logP);

    he.scheme.multByConstAndEqual(c, 0.98, logP);
    he.scheme.reScaleByAndEqual(c, logP);

    he.scheme.addAndEqual(c3, c);

    return c3;
}

void reduceSum(
    HEEnv& he,
    Ciphertext& ct,
    long logSlots,
    CampaignArgs& args
){

    for(int i=0;i<logSlots;i++){
        Ciphertext rot;
        rot = he.scheme.leftRotateFast(ct, 1<<i);
        he.scheme.addAndEqual(ct, rot);

    }
}

vector<Ciphertext> forward(
    HEEnv& he,
    Ciphertext c,
    EncodedWeights& ew,
    long logSlots,
    long logP,
    CampaignArgs& args)
{
    size_t HIDDEN = ew.W1.size();
    size_t OUTPUT = ew.W2.size();

    vector<Ciphertext> layer1(HIDDEN);
    for(size_t j=0;j<HIDDEN;++j){
        Ciphertext s;
            s = he.scheme.multByPoly(c, ew.W1[j], logP);


        he.scheme.reScaleByAndEqual(s, logP);

        reduceSum(he, s, logSlots, args);

        he.scheme.addConstAndEqual(s, ew.b1[j]);

        s = chebyTanh3(he, std::move(s), logP, args);

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


int run_iteration_NN(HEEnv& he, EncodedWeights encoded,
        const vector<double>& vals, CampaignArgs& args, size_t targetValue){

    size_t logSlots = args.logSlots;
    size_t slots = 1 << logSlots;
    size_t logP = args.logDelta;
    size_t logQ = args.logQ;
    size_t verbose = args.verbose;


    vector<complex<double>> arr(slots, {0,0});

    for(size_t i=0;i<vals.size();++i)
        arr[i] = {vals[i],0};

    Plaintext plain = he.scheme.encode(arr.data(), slots, logP, logQ);

    Ciphertext c = he.scheme.encryptMsg(plain, ZZ(args.seed));


    if(args.verbose)
        cout << "Running encrypted inference..." << endl;
    auto outputs = forward(
        he,
        c,
        encoded,
        logSlots,
        logP,
        args
    );

    if(verbose)
        cout << "Decrypting..." << endl;


    auto logitsDec = decryptLogits(he, outputs);



    auto logits = decodeLogits(he, logitsDec);

    size_t pred = 0;
    double best = logits[0];

    for(size_t i=1;i<logits.size();++i){
        if(logits[i] > best){
            best = logits[i];
            pred = i;
        }
    }

    if(verbose){
        cout << "\nPrediction: " << pred
             << "\nTarget:     " << targetValue
             << endl;

        if(pred == targetValue)
            cout << "✔ Correct\n";
        else
            cout << "✘ Incorrect\n";
    } else {
        cout << (pred == targetValue ? 1 : 0) << endl;
    }
    return (pred == targetValue ? 1 : 0);
}

CampaignArgs parse_arguments(int argc, char* argv[]) {
    CampaignArgs args;

    static struct option long_options[] = {
        {"stage",          required_argument, 0, 'S'},
        {"bitPerCoeff",    required_argument, 0, 'c'},
        {"logN",           required_argument, 0, 'N'},
        {"logQ",           required_argument, 0, 'Q'},
        {"logDelta",       required_argument, 0, 'd'},
        {"logSlots",       required_argument, 0, 'g'},
        {"mult_depth",     required_argument, 0, 'm'},
        {"withNTT",        required_argument, 0, 'n'},
        {"doAdd",          required_argument, 0, 'A'},
        {"doPlainMul",     required_argument, 0, 'p'},
        {"doMul",          required_argument, 0, 'M'},
        {"doScalarMul",    required_argument, 0, 'L'},
        {"doRot",          required_argument, 0, 'r'},
        {"doBoot",         required_argument, 0, 'B'},
        {"op_step",        required_argument, 0, 'o'},
        {"op_depth",       required_argument, 0, 'O'},
        {"isComplex",      required_argument, 0, 'X'},
        {"isExhaustive",   required_argument, 0, 'T'},
        {"logMin",         required_argument, 0, 'x'},
        {"logMax",         required_argument, 0, 'y'},
        {"seed",           required_argument, 0, 's'},
        {"seed_input",     required_argument, 0, 'b'},
        // only Openfhe
        {"attackModeSKA",  required_argument, 0, 'a'},
        {"thresholdSKA",   required_argument, 0, 't'},
        {"dnum",           required_argument, 0, 'D'},
        {"amountBits",     required_argument, 0, 'J'},
        {"scaleTech",      required_argument, 0, 'C'},
        {"results_dir",    required_argument, 0, 'R'},
        {"verbose",        no_argument,       0, 'v'},
        {"help",           no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, option_index = 0;

    while ((opt = getopt_long(
        argc, argv,
        "S:c:N:Q:d:g:m:n:A:p:M:L:r:B:o:O:X:T:x:y:s:b:a:t:D:C:R:v:h",
        long_options,
        &option_index)) != -1)
    {
        switch (opt) {
            case 'l':
                args.library = optarg;
                if (args.library != "openfhe" && args.library != "heaan") {
                    std::cerr << "Error: library must be 'openfhe' or 'heaan'\n";
                    std::exit(1);
                }
                break;

            case 'c': args.bitPerCoeff = std::stoul(optarg); break;
            case 'N': args.logN = std::stoul(optarg); break;
            case 'Q': args.logQ = std::stoul(optarg); break;
            case 'd': args.logDelta = std::stoul(optarg); break;
            case 'm': args.mult_depth = std::stoul(optarg); break;
            case 's': args.seed = std::stoul(optarg); break;
            case 'b': args.seed_input = std::stoul(optarg); break;
            case 'x': args.logMin = std::stoul(optarg); break;
            case 'y': args.logMax = std::stoul(optarg); break;
            case 'D': args.dnum= std::stoul(optarg); break;
            case 'r': args.doRot = std::stoul(optarg); break;
            case 'B': args.doBoot = std::stoul(optarg); break;
            case 'o': args.op_step = std::stoul(optarg); break;
            case 'O': args.op_depth = std::stoul(optarg); break;

            case 'v':
                args.verbose = true;
                break;
            case 'g':
                args.logSlots = std::stoul(optarg);
                break;

            case 'n':  // --withNTT 0/1
                args.withNTT = std::stoul(optarg) != 0;
                break;

            case 'A': args.doAdd = std::stoul(optarg); break;
            case 'p': args.doPlainMul = std::stoul(optarg); break;
            case 'M': args.doMul = std::stoul(optarg); break;
            case 'L':
                try {
                    args.doScalarMul = std::stod(optarg);
                } catch (const std::exception& e) {
                    std::cerr << "Invalid value for -L (expected double): " << optarg << "\n";
                    std::exit(EXIT_FAILURE);
                }
                break;

            case 'S':
                args.stage = optarg;
                if (args.stage != "encode" &&
                    args.stage != "encrypt_c0" &&
                    args.stage != "encrypt_c1" &&
                    args.stage != "decrypt_c0" &&
                    args.stage != "decrypt_c1" &&
                    args.stage != "decode" &&
                    args.stage != "cheby_tanh3" &&
                    args.stage != "hidden_layer" &&
                    args.stage != "add_inside" &&
                    args.stage != "mul_inside" &&
                    args.stage != "rescale_inside" &&
                    args.stage != "rot_inside" &&
                    args.stage != "boot_outside" &&
                    args.stage != "boot_coeff" &&
                    args.stage != "boot_eval" &&
                    args.stage != "boot_slot")
                {
                    std::cerr << "Error: invalid stage '" << args.stage
                              << "' (expected: encode, encrypt_c0, encrypt_c1, decrypt_c0, decrypt_c1"
                              " decode, cheby_tanh3, hidden_layer,  mul_inside or mul_outside"
                              "boot_outisde, boot_coeff, boot_eval, boot_slots)\n";
                    std::exit(EXIT_FAILURE);
                }
                break;

            case 'X':
                args.isComplex= std::stoul(optarg);
                break;

            case 'C':
                args.scaleTech= optarg;
                break;

        }
    }

    return args;
}


#include "campaign_helper.h"
#include "campaign_logger.h"
#include "campaign_registry.h"
#include "backend_interface.h"
#include "utils_nn.h"

const size_t INPUT_DIM = 784;
const size_t HIDDEN_DIM = 64;
const size_t OUTPUT_DIM = 10;
const double PIXEL_MAX = 255.0;
const std::string path = "../NN_config/data/";

size_t NUM_BITFLIPS = 50;
ExistingCampaignPolicy existing_policy = ExistingCampaignPolicy::ReuseStrict;

int main(int argc, char* argv[]) {

    std::cout << "\n=== Starting Campaign "<< std::endl;
    CampaignArgs args = parse_arguments(argc, argv);
    args.library = "heaanNN";
    args.isExhaustive= false;
    args.mult_depth = 0;
    args.existing_policy = existing_policy;

    if (args.verbose) {
        args.print();
    }


    long logQ = args.logQ;
    long logP = args.logDelta;
    long logN = args.logN;
    long logSlots = args.logSlots;
    long slots = 1 << logSlots;
    long h = 64;

    size_t targetRow =  args.seed;
    size_t verbose =  args.verbose;

    assert(INPUT_DIM <= slots);
    if(verbose)
        std::cout << "Initializing HE..." << std::endl;

    HEEnv he(logN, logQ, h);

    if(verbose)
        std::cout << "Loading weights..." << std::endl;

    auto W1  = loadCSVMatrix(path+"weights/W1.csv", HIDDEN_DIM, INPUT_DIM);
    auto b1  = loadCSVVector(path+"weights/b1.csv", HIDDEN_DIM);

    auto W2  = loadCSVMatrix(path+"weights/W2.csv", OUTPUT_DIM, HIDDEN_DIM);
    auto b2  = loadCSVVector(path+"weights/b2.csv", OUTPUT_DIM);
    assert(W1[0].size() == INPUT_DIM);
    assert(W2[0].size() == HIDDEN_DIM);



    if(verbose)
        std::cout << "Encoding weights..." << std::endl;

    EncodedWeights encoded =
        encodeWeights(he, W1, b1, W2, b2, slots, logP);

    if(verbose)
        std::cout << "Ready for inference.\n" << std::endl;

    vector<double> vals;
    size_t targetValue;

    bool ok = loadMnistNormRowByIndex(
        path+"mnist_train.csv",
        targetRow,
        targetValue,
        vals
    );

    if(!ok){
        cerr << "Error loading MNIST image\n";
        return 1;
    }


    if(verbose)
        std::cout << "Encrypting input..." << std::endl;
    uint32_t hidden_layer    = 0;
    uint32_t reduceSum_layer = 0;
    IterationResult res = run_iteration_NN(he, encoded, vals, args, targetValue, hidden_layer, reduceSum_layer);
    if(res.detected)
    {
        try{
            args.results_dir = "../../results_NN";
            std::cout << "\n=== Registring Campaign "<< std::endl;
            CampaignRegistry registry(args);

            uint32_t campaign_id = registry.campaign_id;


            std::cout << "\n=== Starting Campaign " << campaign_id << " ===" << std::endl;

            CampaignLogger logger(campaign_id, args.results_dir + "/data", 10000);

            std::cout << "Campaign " << campaign_id << " registered" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            // ========== 10. LOOP DE BIT FLIPS ==========
            std::cout << "\nStarting bit flip campaign..." << std::endl;

            uint32_t N = 1 << args.logN;
            size_t num_bitFlips = NUM_BITFLIPS;
            std::vector<double> norms;
            norms.reserve(num_bitFlips);

            std::cout << "Total bit flips: " << num_bitFlips*14 << std::endl;

            size_t bits_per_coeff = args.bitPerCoeff;

            std::mt19937 rng(args.seed);

           // std::vector<uint32_t> bits_to_flip = extraBitsBetweenDeltaAndQ(args); // 10 values
            std::vector<uint32_t> bits_to_flip = bitsToFlipGenerator(args); // 14 values
            for (size_t bitIndex = 0; bitIndex < bits_to_flip.size() ; bitIndex++) {
                uint32_t bit = bits_to_flip[bitIndex];
                for (size_t i = 0; i < num_bitFlips; i++) {
                    // We already know what happens to all coeffs, so we can reduce the search for the first 8 coeefs
                    uint32_t coeff = random_int(0, (1<<logN)-1);
                    //uint32_t coeff = random_int(0, 8-1);
                    IterationArgs iterArgs(0, coeff, bit);

                    if(args.existing_policy == ExistingCampaignPolicy::Reuse){
                        if (logger.contains(iterArgs))
                        {
                            std::cout << "Skipping already computed iteration\n";
                        }
                    }
                    else{
                        IterationResult res = run_iteration_NN(he, encoded, vals,
                                args, targetValue, hidden_layer,
                                reduceSum_layer, iterArgs);
                        SlotErrorStats  stats;
                        logger.log(iterArgs.limb,
                                iterArgs.coeff,
                                iterArgs.bit,
                                0.0, 0.0,
                                !res.detected,     // is_sdc: predict correct or not, we need to negate. 1 will be sdc, bad. 0 will be mask, good.
                                stats,
                                hidden_layer,
                                reduceSum_layer
                                );
                    }
                }

            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::seconds duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
            uint64_t mins = minutes.count();

            registry.register_end({campaign_id, logger.total(), logger.sdc(), mins, 0.0, 0.0, timestamp_now()});
        }
        catch (const std::runtime_error& e) {
            std::cerr << e.what() << '\n';
            return 0;
        }
    } else {
        std::cout << "Wrong prediction of clean NN" << std::endl;
        return 1;
    }

    return 0;
}

#include "utils_nn.h"
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
size_t LAPS = 15;
using namespace std;

int main(int argc, char* argv[]) {

    std::cout << "\n=== Starting Campaign "<< std::endl;
    CampaignArgs args = parse_arguments(argc, argv);
    args.library = "openfheNN";
    args.isExhaustive= false;

    if (args.verbose) {
        args.print();
    }

    long logQ = args.logQ;
    long logP = args.logDelta;
    long multDepth = args.mult_depth;
    long logN = args.logN;
    long logSlots = args.logSlots;
    long slots = 1 << logSlots;
 //   long h = 64;

    size_t targetRow =  args.seed;
    size_t verbose =  args.verbose;

    assert(INPUT_DIM <= slots);
    if(verbose)
        cout << "Initializing HE..." << endl;

    HEEnv he(logN, multDepth,logP, logQ);

    if(verbose)
        cout << "Loading weights..." << endl;
    auto W1  = loadCSVMatrix(path+"weights/W1.csv", HIDDEN_DIM, INPUT_DIM);
    auto b1  = loadCSVVector(path+"weights/b1.csv", HIDDEN_DIM);

    auto W2  = loadCSVMatrix(path+"weights/W2.csv", OUTPUT_DIM, HIDDEN_DIM);
    auto b2  = loadCSVVector(path+"weights/b2.csv", OUTPUT_DIM);

    assert(W1[0].size() == INPUT_DIM);
    assert(W2[0].size() == HIDDEN_DIM);

    if(verbose)
        cout << "Encoding weights..." << endl;

    EncodedWeights encoded =
        encodeWeights(he, W1, b1, W2, b2, slots);

    if(verbose)
        cout << "Ready for inference.\n" << endl;

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
        cout << "Encrypting input..." << endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    for(size_t i=0 ; i<LAPS;i++){
        args.seed++;
        IterationResult res = run_iteration_NN(he, encoded, vals, args, targetValue);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Time: " << duration.count()/LAPS << " s"  << std::endl;



    return 0;
}

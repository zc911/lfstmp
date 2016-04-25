#include "witness_engine.h"

namespace dg {

WitnessEngine::WitnessEngine(const Config &config) {
    processor_ = NULL;
    is_init_ = false;
    init(config);
}

WitnessEngine::~WitnessEngine() {
    is_init_ = false;

    if (processor_) {
        Processor *next = processor_;
        Processor *to_delete = next;
        do {
            to_delete = next;
            next = next->GetNextProcessor();
            delete to_delete;
            to_delete = NULL;
        } while (next);
    }
}

void WitnessEngine::init(const Config &config) {

    processor_ = new VehicleMultiTypeDetectorProcessor();
    processor_->SetNextProcessor(new VehicleClassifierProcessor());
    is_init_ = true;
}

}

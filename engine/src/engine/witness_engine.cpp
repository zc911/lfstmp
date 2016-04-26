#include "witness_engine.h"
#include "engine_config_value.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"

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

void WitnessEngine::Process(Frame *frame) {

    if (frame->operation().Check(OPERATION_VEHICLE)) {
        vehicle_processor_->Update(frame);
    }

    if (frame->operation().Check(OPERATION_FACE)) {
        face_processor_->Update(frame);
    }

}

// TODO
void WitnessEngine::Process(FrameBatch *frame) {

}

void WitnessEngine::initFeatureOptions(const Config &config) {
    enable_vehicle_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE);

    enable_vehicle_type_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE_TYPE);

    enable_vehicle_color_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE_COLOR);
    enable_vehicle_plate_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE_PLATE);
    enable_vehicle_plate_enhance_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE_PLATE_ENHANCED);
    enable_vehicle_marker_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE_MARKER);
    enable_vehicle_feature_vector_ = (bool) config.Value(
            EngineConfigValue::FEATURE_VEHICLE_ENABLE_FEATURE_VECTOR);

    enable_face_ = (bool) config.Value(EngineConfigValue::FEATURE_FACE_ENABLE);
    enable_face_feature_vector_ = (bool) config.Value(
            FEATURE_FACE_ENABLE_FEATURE_VECTOR);
}

void WitnessEngine::init(const Config &config) {

    if (enable_vehicle_) {
        LOG(INFO)<< "Init vehicle processor pipeline. " << endl;
        vehicle_processor_ = new VehicleMultiTypeDetectorProcessor();
        Processor *last = vehicle_processor_;
        if (enable_vehicle_type_) {
            LOG(INFO)<< "Enable vehicle type classification processor." << endl;
            Processor *vehicle_class = new VehicleClassifierProcessor();
            last->SetNextProcessor(vehicle_class);
            last = vehicle_class;
        }
        if (enable_vehicle_color_) {
            LOG(INFO)<< "Enable vehicle color classification processor." << endl;
        }
        if (enable_vehicle_plate_) {
            LOG(INFO)<< "Enable vehicle plate processor." << endl;
        }
        if (enable_vehicle_marker_) {
            LOG(INFO)<< "Enable vehicle marker processor." << endl;
        }
        if (enable_vehicle_feature_vector_) {
            LOG(INFO)<< "Enable vehicle feature vector processor." << endl;
        }

        LOG(INFO)<< "Init vehicle processor pipeline finished. " << endl;

    }

    if (enable_face_) {
        LOG(INFO) << "Init face processor pipeline. " << endl;
        face_processor_ = new FaceDetectProcessor("","",true, 1, 0.7, 800, 450);
        if(enable_face_feature_vector_) {
            LOG(INFO) << "Enable face feature vector processor." << endl;
            face_processor_->SetNextProcessor(new FaceFeatureExtractProcessor("","", true, 1, "",""));
        }
        LOG(INFO) << "Init face processor pipeline finished. " << endl;
    }

    is_init_ = true;
}

}

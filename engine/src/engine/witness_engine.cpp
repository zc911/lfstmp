#include "witness_engine.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_classifier_processor.h"
#include "processor/vehicle_color_processor.h"
#include "processor/vehicle_marker_classifier_processor.h"
#include "processor/vehicle_plate_recognizer_processor.h"
#include "processor/car_feature_extract_processor.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/config_filter.h"

namespace dg {

WitnessEngine::WitnessEngine(const Config &config) {
    vehicle_processor_ = NULL;
    face_processor_ = NULL;
    is_init_ = false;

    init(config);
}

WitnessEngine::~WitnessEngine() {
    is_init_ = false;

    if (vehicle_processor_) {
        Processor *next = vehicle_processor_;
        Processor *to_delete = next;
        do {
            to_delete = next;
            next = next->GetNextProcessor();
            delete to_delete;
            to_delete = NULL;
        } while (next);
    }

    if (face_processor_) {
        Processor *next = face_processor_;
        Processor *to_delete = next;
        do {
            to_delete = next;
            next = next->GetNextProcessor();
            delete to_delete;
            to_delete = NULL;
        } while (next);
    }
}

void WitnessEngine::Process(FrameBatch *frame) {
    if (frame->CheckFrameBatchOperation(OPERATION_VEHICLE)) {
        if (vehicle_processor_)
            vehicle_processor_->Update(frame);
    }

    if (frame->CheckFrameBatchOperation(OPERATION_FACE)) {
        if (face_processor_)
            face_processor_->Update(frame);
    }
}

void WitnessEngine::initFeatureOptions(const Config &config) {
    enable_vehicle_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE);
    DLOG(INFO) << "begin " << enable_vehicle_ << endl;

    enable_vehicle_type_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_TYPE);

    enable_vehicle_color_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_COLOR);
    enable_vehicle_plate_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_PLATE);
    enable_vehicle_plate_enhance_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_PLATE_ENHANCED);
    enable_vehicle_marker_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_MARKER);
    enable_vehicle_feature_vector_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_FEATURE_VECTOR);

    enable_face_ = (bool) config.Value(FEATURE_FACE_ENABLE);
    enable_face_feature_vector_ = (bool) config.Value(
        FEATURE_FACE_ENABLE_FEATURE_VECTOR);

}

void WitnessEngine::init(const Config &config) {

    ConfigFilter *configFilter = ConfigFilter::GetInstance();
    if (!configFilter->initDataConfig(config)) {
        LOG(ERROR) << "can not init data config" << endl;
        DLOG(ERROR) << "can not init data config" << endl;
        return;
    }
    initFeatureOptions(config);
    if (enable_vehicle_) {
        LOG(INFO) << "Init vehicle processor pipeline. " << endl;

        VehicleCaffeDetector::VehicleCaffeDetectorConfig dConfig;
        configFilter->createVehicleCaffeDetectorConfig(config, dConfig);
        vehicle_processor_ = new VehicleMultiTypeDetectorProcessor(dConfig);
        Processor *last = vehicle_processor_;

        if (enable_vehicle_type_) {
            LOG(INFO) << "Enable vehicle type classification processor." << endl;
            vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs;
            configFilter->createVehicleConfig(config, configs);

            Processor *p = new VehicleClassifierProcessor(configs);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_color_) {
            LOG(INFO) << "Enable vehicle color classification processor." << endl;
            vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs;
            configFilter->createVehicleColorConfig(config, configs);

            Processor *p = new VehicleColorProcessor(configs);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_plate_) {
            LOG(INFO) << "Enable vehicle plate processor." << endl;
            PlateRecognizer::PlateConfig pConfig;
            configFilter->createVehiclePlateConfig(config, pConfig);
            Processor *p = new PlateRecognizerProcessor(pConfig);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_marker_) {
            LOG(INFO) << "Enable vehicle marker processor." << endl;
            MarkerCaffeClassifier::MarkerConfig mConfig;
            configFilter->createMarkersConfig(config, mConfig);
            WindowCaffeDetector::WindowCaffeConfig wConfig;
            configFilter->createWindowConfig(config, wConfig);

            Processor *p = new VehicleMarkerClassifierProcessor(wConfig, mConfig);
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_feature_vector_) {
            LOG(INFO) << "Enable vehicle feature vector processor." << endl;
            Processor *p = new CarFeatureExtractProcessor();
            last->SetNextProcessor(p);
            last = p;
        }

        LOG(INFO) << "Init vehicle processor pipeline finished. " << endl;

    }

    if (enable_face_) {
        LOG(INFO) << "Init face processor pipeline. " << endl;
        FaceDetector::FaceDetectorConfig fdconfig;
        configFilter->createFaceDetectorConfig(config, fdconfig);
        face_processor_ = new FaceDetectProcessor(fdconfig);

        if (enable_face_feature_vector_) {
            LOG(INFO) << "Enable face feature vector processor." << endl;
            FaceFeatureExtractor::FaceFeatureExtractorConfig feconfig;
            configFilter->createFaceExtractorConfig(config, feconfig);
            face_processor_->SetNextProcessor(new FaceFeatureExtractProcessor(feconfig));
        }
        LOG(INFO) << "Init face processor pipeline finished. " << endl;
    }

    is_init_ = true;
}

}

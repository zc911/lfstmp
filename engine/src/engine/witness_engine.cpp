#include "witness_engine.h"


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
    DLOG(INFO)<<"begin "<<enable_vehicle_<<endl;

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

    initFeatureOptions(config);
    if (enable_vehicle_) {
        LOG(INFO)<< "Init vehicle processor pipeline. " << endl;

        vehicle_processor_ = new VehicleMultiTypeDetectorProcessor(1, 0, 600,false);
        Processor *last = vehicle_processor_;

        if (enable_vehicle_type_) {
            LOG(INFO)<< "Enable vehicle type classification processor." << endl;
            DLOG(INFO)<<"begin  "<<endl;
            vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs = createVehicleConfig(config);
            Processor *p = new VehicleClassifierProcessor(configs);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_color_) {
            LOG(INFO)<< "Enable vehicle color classification processor." << endl;
            vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs = createVehicleColorConfig(config);

            Processor *p = new VehicleColorProcessor(configs);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_plate_) {
            LOG(INFO)<< "Enable vehicle plate processor." << endl;
            PlateRecognizer::PlateConfig pConfig;

            Processor *p = new PlateRecognizerProcessor(pConfig);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_marker_) {
            LOG(INFO)<< "Enable vehicle marker processor." << endl;
            Processor *p = new VehicleMarkerClassifierProcessor();
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_feature_vector_) {
            LOG(INFO)<< "Enable vehicle feature vector processor." << endl;
            Processor *p = new CarFeatureExtractProcessor();
            last->SetNextProcessor(p);
            last = p;
        }

        LOG(INFO)<< "Init vehicle processor pipeline finished. " << endl;

    }

    if (enable_face_) {
        LOG(INFO)<< "Init face processor pipeline. " << endl;
        // face_processor_ = new FaceDetectProcessor("","",true, 1, 0.7, 800, 450);
        if(enable_face_feature_vector_) {
            //LOG(INFO) << "Enable face feature vector processor." << endl;
            //face_processor_->SetNextProcessor(new FaceFeatureExtractProcessor("","", true, 1, "",""));
        }
        LOG(INFO) << "Init face processor pipeline finished. " << endl;
    }

    is_init_ = true;
}
const vector<VehicleCaffeClassifier::VehicleCaffeConfig> & WitnessEngine::createVehicleConfig(
        const Config &config) {
    vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs;
    for (int i = 0; i < 8; i++) {
        VehicleCaffeClassifier::VehicleCaffeConfig config;
        config.model_file = "models/car_style/front_day_" + to_string(i)
                + "/car_python_mini_alex_256_" + to_string(i)
                + "_iter_70000.caffemodel";
        config.deploy_file = "models/car_style/front_day_" + to_string(i)
                + "/deploy_256.prototxt";
        config.is_model_encrypt = false;
        config.batch_size = 1;

        configs.push_back(config);
    }
}
const vector<VehicleCaffeClassifier::VehicleCaffeConfig> &WitnessEngine::createVehicleColorConfig(
        const Config &config) {
    vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs;
    for (int i = 0; i < 1; i++) {
        VehicleCaffeClassifier::VehicleCaffeConfig config;
        config.model_file = "models/color/zf_q_iter_70000.caffemodel";
        config.deploy_file = "models/color/deploy.prototxt";
        config.is_model_encrypt = false;
        config.batch_size = 1;

        configs.push_back(config);
    }
    return configs;
}
const PlateRecognizer::PlateConfig& WitnessEngine::createVehiclePlateConfig(
        const Config &config) {
    PlateRecognizer::PlateConfig pConfig;
    pConfig.LocalProvince = "";
    pConfig.OCR = 1;
    pConfig.PlateLocate = 5;
    return pConfig;
}
}

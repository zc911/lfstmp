#include "witness_engine.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_classifier_processor.h"
#include "processor/vehicle_color_processor.h"
#include "processor/vehicle_marker_classifier_processor.h"
#include "processor/vehicle_belt_classifier_processor.h"
#include "processor/non_motor_vehicle_classifier_processor.h"
#include "processor/vehicle_plate_recognizer_processor.h"
#include "processor/car_feature_extract_processor.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/vehicle_window_detector_processor.h"
#include "processor/face_quality_processor.h"
#include "processor/face_alignment_processor.h"
#include "processor/plate_recognize_mxnet_processor.h"
#include "processor/pedestrian_classifier_processor.h"
#include "processor/vehicle_phone_detector_processor.h"
#include "processor/config_filter.h"
#include "debug_util.h"
#include "algorithm_factory.h"
#include "engine_config_value.h"

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

void WitnessEngine::withoutDetection(FrameBatch *frames, Operations oprations) {

    Identification baseid = 0;
    for (auto frame: frames->frames()) {
        Operation op = frame->operation();
        Mat tmp = frame->payload()->data();

        if (tmp.empty()) {
            LOG(ERROR) << "Mat is empty" << endl;
            continue;
        }

        Detection d;
        d.set_box(Rect(0, 0, tmp.cols, tmp.rows));
        Object *obj;

        if (oprations == OPERATION_VEHICLE_DETECT) {
            if (op.Check(OPERATION_PEDESTRIAN_ATTR)) {
                obj = new Pedestrian();
            } else if (op.Check(OPERATION_NON_VEHICLE_ATTR)) {
                obj = new NonMotorVehicle(OBJECT_BICYCLE);
            } else if (op.Check(
                OPERATION_VEHICLE_STYLE | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER | OPERATION_VEHICLE_PLATE
                    | OPERATION_VEHICLE_FEATURE_VECTOR | OPERATION_DRIVER_BELT | OPERATION_CODRIVER_BELT
                    | OPERATION_DRIVER_PHONE)) {
                obj = new Vehicle(OBJECT_CAR);
                // set pose head in default
                Vehicle *v = (Vehicle *) obj;
                v->set_pose(Vehicle::VEHICLE_POSE_HEAD);
            } else {
                continue;
            }
        } else if (oprations == OPERATION_FACE_DETECT) {
            if (op.Check(OPERATION_FACE_FEATURE_VECTOR | OPERATION_FACE_ALIGNMENT | OPERATION_FACE_QUALITY)) {
                obj = new Face();
                Face *f = (Face *) obj;
                f->set_full_image(tmp);
            } else {
                continue;
            }
        } else {
            continue;
        }

        obj->set_id(baseid++);
        obj->set_image(tmp);
        obj->set_detection(d);
        frame->put_object(obj);
    }

}


void WitnessEngine::Process(FrameBatch *frames) {
    float costtime, diff;
    struct timeval start, end;
    gettimeofday(&start, NULL);

    performance_ += frames->frames().size();
#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordPerformance(FEATURE_RESERVED,  performance_)) {
            return;
        }
    }
#endif
    VLOG(VLOG_RUNTIME_DEBUG) << "Start witness engine process" << endl;

    if (!frames->CheckFrameBatchOperation(OPERATION_VEHICLE_DETECT)) {
        withoutDetection(frames, OPERATION_VEHICLE_DETECT);
    }

    if (!frames->CheckFrameBatchOperation(OPERATION_FACE_DETECT)) {
        withoutDetection(frames, OPERATION_FACE_DETECT);
    }

    if (frames->CheckFrameBatchOperation(OPERATION_VEHICLE)) {
        if (vehicle_processor_) {
            vehicle_processor_->Update(frames);
        }
    }
    if (frames->CheckFrameBatchOperation(OPERATION_FACE)) {
        if (face_processor_)
            face_processor_->Update(frames);
    }

    gettimeofday(&end, NULL);

    int cost = TimeCostInMs(start, end);
    VLOG(VLOG_RUNTIME_DEBUG) << " [witness engine cost]: " << cost;

}

void WitnessEngine::initFeatureOptions(const Config &config) {
    enable_vehicle_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE);
    enable_face_ = (bool) config.Value(FEATURE_FACE_ENABLE);
    enable_non_motor_vehicle_ = (bool) config.Value(FEATURE_NON_MOTOR_VEHICLE_ENABLE);

#if DEBUG
    enable_vehicle_detect_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_DETECTION);
    enable_vehicle_type_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_TYPE);

    enable_vehicle_color_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_COLOR);
    enable_vehicle_plate_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_PLATE);
    enable_vehicle_plate_gpu_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_GPU_PLATE);

    enable_vehicle_marker_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_MARKER);
    enable_vehicle_feature_vector_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_FEATURE_VECTOR);
    enable_vehicle_pedestrian_attr_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_PEDISTRIAN_ATTR);

    enable_face_feature_vector_ = (bool) config.Value(
        FEATURE_FACE_ENABLE_FEATURE_VECTOR);
    enable_face_detect_ = (bool) config.Value(
        FEATURE_FACE_ENABLE_DETECTION);
    enable_face_quality_ = (bool) config.Value(FEATURE_FACE_ENABLE_QUALITY);
    enable_face_alignment_ = (bool) config.Value(FEATURE_FACE_ENABLE_ALIGNMENT);
    enable_face_pose_ = (bool) config.Value(FEATURE_FACE_ENABLE_POSE);


    enable_vehicle_driver_belt_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_DRIVERBELT);
    enable_vehicle_codriver_belt_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_CODRIVERBELT);
    enable_vehicle_driver_phone_ = (bool) config.Value(
        FEATURE_VEHICLE_ENABLE_PHONE);

#else
    enable_vehicle_detect_ = (bool) config.Value(
                                 FEATURE_VEHICLE_ENABLE_DETECTION) && (CheckFeature(FEATURE_CAR_DETECTION, false) == ERR_FEATURE_ON);
    enable_vehicle_type_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_TYPE) && (CheckFeature(FEATURE_CAR_STYLE, false) == ERR_FEATURE_ON);

    enable_vehicle_color_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_COLOR) && (CheckFeature(FEATURE_CAR_COLOR, false) == ERR_FEATURE_ON);
    enable_vehicle_plate_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_PLATE) && (CheckFeature(FEATURE_CAR_PLATE, false) == ERR_FEATURE_ON);
    enable_vehicle_plate_gpu_ = (bool) config.Value(
                                    FEATURE_VEHICLE_ENABLE_GPU_PLATE) && (CheckFeature(FEATURE_CAR_PLATE, false) == ERR_FEATURE_ON);

    enable_vehicle_marker_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE_MARKER) && (CheckFeature(FEATURE_CAR_MARK, false) == ERR_FEATURE_ON);
    enable_vehicle_feature_vector_ = (bool) config.Value(
                                         FEATURE_VEHICLE_ENABLE_FEATURE_VECTOR) && (CheckFeature(FEATURE_CAR_EXTRACT, false) == ERR_FEATURE_ON);
    enable_vehicle_pedestrian_attr_ = (bool) config.Value(
                                          FEATURE_VEHICLE_ENABLE_PEDISTRIAN_ATTR) && (CheckFeature(FEATURE_CAR_PEDESTRIAN_ATTR, false) == ERR_FEATURE_ON);

    enable_face_detect_ = (bool) config.Value(
                              FEATURE_FACE_ENABLE_FEATURE_VECTOR) && (CheckFeature(FEATURE_FACE_DETECTION, false) == ERR_FEATURE_ON);
    enable_face_quality_ = (bool) config.Value( FEATURE_FACE_ENABLE_QUALITY);
    enable_face_pose_ = (bool) config.Value( FEATURE_FACE_ENABLE_POSE);

    enable_face_feature_vector_ = (bool) config.Value(
                                      FEATURE_FACE_ENABLE_FEATURE_VECTOR) && (CheckFeature(FEATURE_FACE_EXTRACT, false) == ERR_FEATURE_ON);
    enable_vehicle_driver_belt_ = (bool) config.Value(
                                      FEATURE_VEHICLE_ENABLE_DRIVERBELT) && (CheckFeature(FEATURE_CAR_MARK, false) == ERR_FEATURE_ON);
    enable_vehicle_codriver_belt_ = (bool) config.Value(
                                        FEATURE_VEHICLE_ENABLE_CODRIVERBELT) && (CheckFeature(FEATURE_CAR_BEHAVIOR_PHONE, false) == ERR_FEATURE_ON);
    enable_vehicle_driver_phone_ = (bool) config.Value(
                                       FEATURE_VEHICLE_ENABLE_PHONE) && (CheckFeature(FEATURE_CAR_BEHAVIOR_NOBELT, false) == ERR_FEATURE_ON);
#endif

}

void WitnessEngine::recordPerformance() {

}

void WitnessEngine::init(const Config &config) {

    int gpu_id = (bool) config.Value(SYSTEM_GPUID);
    bool is_encrypted = (bool) config.Value(DEBUG_MODEL_ENCRYPT);
    string dgvehiclePath = (string) config.Value(DGVEHICLE_MODEL_PATH);
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize(dgvehiclePath, gpu_id, is_encrypted);

    ConfigFilter *configFilter = ConfigFilter::GetInstance();
    if (!configFilter->initDataConfig(config)) {
        LOG(ERROR) << "can not init data config" << endl;
        DLOG(ERROR) << "can not init data config" << endl;
        return;
    }

    initFeatureOptions(config);

    Processor *last = NULL;
    if (enable_vehicle_) {
        LOG(INFO) << "Init vehicle processor pipeline. " << endl;
        LOG(INFO) << "Enable accelerate detection processor." << endl;

        bool car_only = (bool) config.Value(ADVANCED_DETECTION_CAR_ONLY);
        Processor *p = new VehicleMultiTypeDetectorProcessor(car_only, true);
        vehicle_processor_ = p;
        last = p;

        if (enable_vehicle_detect_) {
            LOG(INFO) << "Enable  detection processor." << endl;

            p = new VehicleMultiTypeDetectorProcessor(car_only, false);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_plate_gpu_) {
            LOG(INFO) << "Enable plate detection processor." << endl;

            PlateRecognizeMxnetProcessor::PlateRecognizeMxnetConfig pConfig;
            configFilter->createPlateMxnetConfig(config, pConfig);

            p = new PlateRecognizeMxnetProcessor(pConfig);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_type_) {
            LOG(INFO) << "Enable vehicle type classification processor." << endl;

            string mappingFilePath = (string) config.Value("Render/Vehicle/Model");
            p = new VehicleClassifierProcessor(mappingFilePath);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_color_) {
            LOG(INFO) << "Enable vehicle color classification processor." << endl;

            p = new VehicleColorProcessor();
            last->SetNextProcessor(p);
            last = p;
        }

        // We disable the TH plate sdk mode
//        if (enable_vehicle_plate_ & 0) {
//            LOG(INFO) << "Enable vehicle plate processor." << endl;
//            PlateRecognizer::PlateConfig pConfig;
//            configFilter->createVehiclePlateConfig(config, pConfig);
//            Processor *p = new PlateRecognizerProcessor(pConfig);
//            if (last == NULL) {
//                vehicle_processor_ = p;
//            }
//            else {
//                last->SetNextProcessor(p);
//            }
//            last = p;
//        }

        if (enable_vehicle_marker_ || enable_vehicle_driver_belt_ || enable_vehicle_codriver_belt_
            || enable_vehicle_driver_phone_) {
            LOG(INFO) << "Enable vehicle window processor." << endl;

            p = new VehicleWindowDetectorProcessor();
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_marker_) {
            LOG(INFO) << "Enable vehicle marker processor." << endl;
            p = new VehicleMarkerClassifierProcessor(false);
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_driver_belt_) {
            LOG(INFO) << "Enable vehicle driver belt processor." << endl;
            float threshold = (float) config.Value(ADVANCED_DRIVER_BELT_THRESHOLD);
            p = new VehicleBeltClassifierProcessor(threshold, true);
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_codriver_belt_) {
            LOG(INFO) << "Enable vehicle co-driver belt processor." << endl;
            float threshold = (float) config.Value(ADVANCED_CODRIVER_BELT_THRESHOLD);
            p = new VehicleBeltClassifierProcessor(threshold, false);
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_driver_phone_) {
            LOG(INFO) << "Enable vehicle driver phone processor." << endl;
            float threshold = (float) config.Value(ADVANCED_PHONE_THRESHOLD);
            p = new VehiclePhoneClassifierProcessor(threshold);
            last->SetNextProcessor(p);
            last = p;
        }
        if (enable_vehicle_feature_vector_) {
            LOG(INFO) << "Enable vehicle feature vector processor." << endl;
            p = new CarFeatureExtractProcessor();
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_vehicle_pedestrian_attr_) {
            LOG(INFO) << "Enable vehicle pedestrian attr processor." << endl;
            p = new PedestrianClassifierProcessor();
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_non_motor_vehicle_) {
            LOG(INFO) << "Enable non-motor vehicle attribute processor" << endl;
            p = new NonMotorVehicleClassifierProcessor();
            last->SetNextProcessor(p);
            last = p;
        }

        LOG(INFO) << "Init vehicle processor pipeline finished. " << endl;
        last->SetNextProcessor(NULL);
    }

    if (enable_face_) {
        LOG(INFO) << "Init face processor pipeline. " << endl;
        unsigned int method = (int) config.Value(ADVANCED_FACE_DETECT_METHOD);

        VLOG(VLOG_RUNTIME_DEBUG) << "Start load face detection model" << endl;
        FaceDetectorConfig fdconfig;
        configFilter->createFaceDetectorConfig(config, fdconfig);
        face_processor_ = new FaceDetectProcessor(fdconfig, (FaceDetectProcessor::FaceDetectMethod)method);
        Processor *last = face_processor_;

        if (enable_face_alignment_) {
            LOG(INFO) << "Enable face alignment processor." << endl;
            VLOG(VLOG_RUNTIME_DEBUG) << "Start load face alignment model" << endl;
            FaceAlignmentConfig faConfig;
            configFilter->createFaceAlignmentConfig(config, faConfig);
            Processor *p = new FaceAlignmentProcessor(faConfig, (FaceDetectProcessor::FaceDetectMethod)method);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_face_quality_) {
            LOG(INFO) << "Enable face quality processor." << endl;
            VLOG(VLOG_RUNTIME_DEBUG) << "Start load face quality model" << endl;
            FaceQualityConfig fqConfig;
            configFilter->createFaceQualityConfig(config, fqConfig);
            Processor *p = new FaceQualityProcessor(fqConfig);
            last->SetNextProcessor(p);
            last = p;
        }

        if (enable_face_feature_vector_) {
            LOG(INFO) << "Enable face feature vector processor." << endl;
            VLOG(VLOG_RUNTIME_DEBUG) << "Start load face feature extract model" << endl;
            FaceFeatureExtractorConfig feconfig;
            FaceAlignmentConfig faConfig;
            configFilter->createFaceExtractorConfig(config, feconfig);
            Processor *p = new FaceFeatureExtractProcessor(feconfig);
            last->SetNextProcessor(p);
            last = p;
        }
//        if (enable_face_feature_vector_) {
//            LOG(INFO) << "Enable face pose processor." << endl;
//            VLOG(VLOG_RUNTIME_DEBUG) << "Start load face pose model" << endl;
//            FacePoseConfig fpconfig;
//            Processor *p = new FacePoseProcessor(fpconfig);
//            last->SetNextProcessor(p);
//            last = p;
//        }
        last->SetNextProcessor(NULL);


        LOG(INFO) << "Init face processor pipeline finished. " << endl;
    }
    if (!RecordPerformance(FEATURE_RESERVED, performance_)) {
        performance_ = RECORD_UNIT;
    }

    Mat image = Mat::zeros(100, 100, CV_8UC3);
    FrameBatch framebatch(0);
    Frame *frame = new Frame(0, image);
    framebatch.AddFrame(frame);
    this->Process(&framebatch);

    if (vehicle_processor_)
        vehicle_processor_ = vehicle_processor_->GetNextProcessor();

    this->Process(&framebatch);

    is_init_ = true;

}

void WitnessEngine::initGpuMemory(FrameBatch &batch) {

    Mat image = Mat::zeros(1000, 1000, CV_8UC3);
    Mat smallImage = Mat::zeros(50, 50, CV_8UC3);
    Operation op;
    op.Set(1023);
    for (int i = 0; i < 16; ++i) {
        Frame *frame = new Frame(i, image);
        Vehicle *vehicle = new Vehicle(OBJECT_CAR);
        vehicle->set_id(1);
        vehicle->set_image(smallImage);
        vector<Detection> markers;
        Detection det;
        det.set_box(cv::Rect(1, 1, 10, 10));
        markers.push_back(det);
        //vehicle->set_markers(markers);
        Pedestrian *pedestrain = new Pedestrian();
        pedestrain->set_image(smallImage);
        pedestrain->set_id(2);
        frame->put_object(vehicle);
        frame->put_object(pedestrain);
        frame->set_operation(op);
        batch.AddFrame(frame);
    }
}

}

#include "witness_engine.h"
#include "log/log_val.h"
#include "processor/vehicle_multi_type_detector_processor.h"
#include "processor/vehicle_classifier_processor.h"
#include "processor/vehicle_color_processor.h"
#include "processor/vehicle_marker_classifier_processor.h"
#include "processor/vehicle_plate_recognizer_processor.h"
#include "processor/car_feature_extract_processor.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/config_filter.h"
#include "processor/plate_recognize_mxnet_processor.h"
#include "processor/pedestrian_classifier_processor.h"

namespace dg
{

WitnessEngine::WitnessEngine(const Config &config)
{
	vehicle_processor_ = NULL;
	face_processor_ = NULL;
	is_init_ = false;

	init(config);
}

WitnessEngine::~WitnessEngine()
{
	is_init_ = false;

	if (vehicle_processor_)
	{
		Processor *next = vehicle_processor_;
		Processor *to_delete = next;
		do
		{
			to_delete = next;
			next = next->GetNextProcessor();
			delete to_delete;
			to_delete = NULL;
		} while (next);
	}

	if (face_processor_)
	{
		Processor *next = face_processor_;
		Processor *to_delete = next;
		do
		{
			to_delete = next;
			next = next->GetNextProcessor();
			delete to_delete;
			to_delete = NULL;
		} while (next);
	}
}

void WitnessEngine::Process(FrameBatch *frames)
{

	VLOG(VLOG_RUNTIME_DEBUG) << "Start witness engine process" << endl;

	if (frames->CheckFrameBatchOperation(OPERATION_VEHICLE))
	{
		if (!enable_vehicle_detect_
				|| !frames->CheckFrameBatchOperation(OPERATION_VEHICLE_DETECT))
		{

			if (frames->CheckFrameBatchOperation(OPERATION_VEHICLE_PEDESTRIAN_ATTR))
			{
				Identification baseid = 0;
				for (auto frame : frames->frames())
				{
					Pedestrian *p = new Pedestrian();
					Mat tmp = frame->payload()->data();
					p->set_image(tmp);
					p->set_id(baseid);
					baseid++;
					Object *obj = static_cast<Object *>(p);
					if (obj)
					{
						Detection d;
						d.box = Rect(0, 0, tmp.cols, tmp.rows);
						obj->set_detection(d);
						frame->put_object(obj);
					}
				}
			}
			else
			{
				Identification baseid = 0;
				for (auto frame : frames->frames())
				{
					Vehicle *v = new Vehicle(OBJECT_CAR);
					Mat tmp = frame->payload()->data();
					v->set_image(tmp);
					v->set_id(baseid);
					baseid++;
					Object *obj = static_cast<Object *>(v);

					if (obj)
					{
						Detection d;
						d.box = Rect(0, 0, tmp.cols, tmp.rows);
						obj->set_detection(d);
						frame->put_object(obj);
					}
				}
			}
		}

		if (vehicle_processor_)
			vehicle_processor_->Update(frames);
	}

	if (frames->CheckFrameBatchOperation(OPERATION_FACE))
	{
		if (face_processor_)
			face_processor_->Update(frames);
	}
	if (!isWarmuped_ && ((!enable_vehicle_) || (!enable_vehicle_detect_)))
	{
		vehicle_processor_ = vehicle_processor_->GetNextProcessor();
		isWarmuped_ = true;
	}
}

void WitnessEngine::initFeatureOptions(const Config &config)
{
	enable_vehicle_ = (bool) config.Value(FEATURE_VEHICLE_ENABLE);

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

	enable_face_ = (bool) config.Value(FEATURE_FACE_ENABLE);
	enable_face_feature_vector_ = (bool) config.Value(
			FEATURE_FACE_ENABLE_FEATURE_VECTOR);

}

void WitnessEngine::recordPerformance()
{

}
void WitnessEngine::init(const Config &config)
{

	ConfigFilter *configFilter = ConfigFilter::GetInstance();
	if (!configFilter->initDataConfig(config))
	{
		LOG(ERROR)<< "can not init data config" << endl;
		DLOG(ERROR) << "can not init data config" << endl;
		return;
	}

	initFeatureOptions(config);

	Processor *last = NULL;
	if (enable_vehicle_)
	{
		LOG(INFO)<< "Init vehicle processor pipeline. " << endl;

		if (enable_vehicle_detect_)
		{

			CarOnlyCaffeDetector::VehicleCaffeDetectorConfig dConfig;
			configFilter->createVehicleCaffeDetectorConfig(config, dConfig);
			Processor *p = new VehicleMultiTypeDetectorProcessor(dConfig);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}
		else
		{
			CarOnlyCaffeDetector::VehicleCaffeDetectorConfig dConfig;
			configFilter->createAccelerateConfig(config, dConfig);
			Processor *p = new VehicleMultiTypeDetectorProcessor(dConfig);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
			isWarmuped_=false;
		}

		if (enable_vehicle_plate_gpu_)
		{
			LPDRConfig_S pstConfig;
			configFilter->createPlateMxnetConfig(config, &pstConfig);

			Processor *p = new PlateRecognizeMxnetProcessor(&pstConfig);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		if (enable_vehicle_type_)
		{
			LOG(INFO) << "Enable vehicle type classification processor." << endl;
			vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs;
			configFilter->createVehicleConfig(config, configs);

			Processor *p = new VehicleClassifierProcessor(configs);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		if (enable_vehicle_color_)
		{
			LOG(INFO) << "Enable vehicle color classification processor." << endl;
			vector<CaffeVehicleColorClassifier::VehicleColorConfig> configs;
			configFilter->createVehicleColorConfig(config, configs);

			Processor *p = new VehicleColorProcessor(configs);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		if (enable_vehicle_plate_)
		{
			LOG(INFO) << "Enable vehicle plate processor." << endl;
			PlateRecognizer::PlateConfig pConfig;
			configFilter->createVehiclePlateConfig(config, pConfig);
			Processor *p = new PlateRecognizerProcessor(pConfig);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		if (enable_vehicle_marker_)
		{
			LOG(INFO) << "Enable vehicle marker processor." << endl;
			MarkerCaffeClassifier::MarkerConfig mConfig;
			configFilter->createMarkersConfig(config, mConfig);
			WindowCaffeDetector::WindowCaffeConfig wConfig;
			configFilter->createWindowConfig(config, wConfig);

			Processor *p = new VehicleMarkerClassifierProcessor(wConfig, mConfig);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		if (enable_vehicle_feature_vector_)
		{
			LOG(INFO) << "Enable vehicle feature vector processor." << endl;
			Processor *p = new CarFeatureExtractProcessor();
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		if(enable_vehicle_pedestrian_attr_)
		{
			LOG(INFO) << "Enable vehicle pedestrian attr processor." << endl;
			PedestrianClassifier::PedestrianConfig pConfig;
			configFilter->createPedestrianConfig(config, pConfig);
			Processor *p = new PedestrianClassifierProcessor(pConfig);
			if (last == NULL)
			{
				vehicle_processor_ = p;
			}
			else
			{
				last->SetNextProcessor(p);
			}
			last = p;
		}

		LOG(INFO) << "Init vehicle processor pipeline finished. " << endl;
		last->SetNextProcessor(NULL);
	}

	if (enable_face_)
	{
		LOG(INFO)<< "Init face processor pipeline. " << endl;
		FaceDetector::FaceDetectorConfig fdconfig;
		configFilter->createFaceDetectorConfig(config, fdconfig);
		face_processor_ = new FaceDetectProcessor(fdconfig);

		if (enable_face_feature_vector_)
		{
			LOG(INFO) << "Enable face feature vector processor." << endl;
			FaceFeatureExtractor::FaceFeatureExtractorConfig feconfig;
			configFilter->createFaceExtractorConfig(config, feconfig);
			face_processor_->SetNextProcessor(new FaceFeatureExtractProcessor(feconfig));
		}

		LOG(INFO) << "Init face processor pipeline finished. " << endl;
	}

	/*  Mat image = Mat::zeros(480,480,CV_8UC3);
	 Frame *f = new Frame(100, image);
	 Operation op;
	 op.Set(OPERATION_VEHICLE);
	 op.Set(OPERATION_VEHICLE_DETECT | OPERATION_VEHICLE_STYLE
	 | OPERATION_VEHICLE_COLOR | OPERATION_VEHICLE_MARKER
	 | OPERATION_VEHICLE_FEATURE_VECTOR
	 | OPERATION_VEHICLE_PLATE);
	 op.Set(OPERATION_FACE | OPERATION_FACE_DETECTOR
	 | OPERATION_FACE_FEATURE_VECTOR);

	 f->set_operation(op);
	 FrameBatch *fb = new FrameBatch(1);
	 fb->AddFrame(f);
	 this->Process(fb);
	 delete fb;*/
	// vehicle_processor_=vehicle_processor_->GetNextProcessor();
	is_init_ = true;
}

}

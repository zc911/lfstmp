#include "processor/face_quality_processor.h"

#include "processor_helper.h"
namespace dg {

FaceQualityProcessor::FaceQualityProcessor( const FaceQualityConfig &config) {
	switch (config.frontalMethod) {
	case FrontalDlib:
		fq_ = new DGFace::FrontalMQuality();
		frontalThreshold_ = config.frontalThreshold;
		break;

	}

}
FaceQualityProcessor::~FaceQualityProcessor() {

	delete fq_;
}
bool FaceQualityProcessor::process(FrameBatch *frameBatch) {
	if (!fq_)
		return false;

	for (vector<Object *>::iterator itr = to_processed_.begin(); itr != to_processed_.end();) {
		Mat img = ((Face *)(*itr))->image();
		float score = fq_->quality(img((((Face*)(*itr))->detection()).box));
		if (frontalThreshold_ > score) {
			((Face *)(*itr))->set_valid(false);

			itr = to_processed_.erase(itr);
		} else {
			((Face *)(*itr))->set_qualities(Face::Frontal, score);
			itr++;
		}
		performance_++;
	}
	for (auto *frame : frameBatch->frames()) {
		frame->DeleteInvalidObjects();
	}
	return true;
}
bool FaceQualityProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else
	if (performance_ > RECORD_UNIT) {
		if (!RecordFeaturePerformance()) {
			return false;
		}
	}
#endif
	to_processed_.clear();
	to_processed_ = frameBatch->CollectObjects(OPERATION_FACE_FEATURE_VECTOR);
	for (vector<Object *>::iterator itr = to_processed_.begin();
	        itr != to_processed_.end();) {
		if ((*itr)->type() != OBJECT_FACE) {
			itr = to_processed_.erase(itr);
		} else {
			itr++;

		}
	}
	return true;
}
bool FaceQualityProcessor::RecordFeaturePerformance() {

	return RecordPerformance(FEATURE_FACE_DETECTION, performance_);

}
}
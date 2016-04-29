#include "rank_engine.h"
//#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"
namespace dg
{

//CarRankEngine::CarRankEngine()
//        : id_(0) {
//    processor_ = new CarRankProcessor();
//}
//CarRankEngine::~CarRankEngine() {
//    if (processor_) {
//        delete processor_;
//    }
//}

//vector<Score> CarRankEngine::Rank(const Mat& image, const Rect& hotspot,
//                                  const vector<CarRankFeature>& candidates) {
//    vector<Rect> hotspots;
//    hotspots.push_back(hotspot);
//    CarRankFrame f(id_++, image, hotspots, candidates);
//    processor_->Update(&f);
//    return f.result_;
//}

FaceRankEngine::FaceRankEngine() :
		id_(0)
{
	detector_ = new FaceDetectProcessor("/models/test.prototxt",
			"/models/googlenet_face_iter_100000.caffemodel", true, 1, 0.7, 800,
			450);

	extractor_ = new FaceFeatureExtractProcessor("/models/lcnn.prototxt",
			"/models/lcnn.caffemodel", true, 1,
			"/models/shape_predictor_68_face_landmarks.dat",
			"/models/avgface.jpg");
}

FaceRankEngine::~FaceRankEngine()
{
	delete detector_;
	delete extractor_;
}

vector<Score> FaceRankEngine::Rank(const Mat& image, const Rect& hotspot,
		const vector<FaceRankFeature>& candidates)
{

	Frame *frame = new Frame(0, image);
	detector_->Update(frame);
	extractor_->Update(frame);

	Face *face = (Face *) frame->get_object(0);
	FaceRankFeature feature = face->feature();
	vector<Rect> hotspots;

	FaceRankFrame *face_rank_frame = new FaceRankFrame(0, feature, hotspots,
			candidates);
	FaceRankProcessor *ranker = new FaceRankProcessor();
	ranker->Update(face_rank_frame);

	vector<Score> result = face_rank_frame->result_;
	delete frame;
	delete face_rank_frame;

	return result;
}

}

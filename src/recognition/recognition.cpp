#include <recognition.h>

using namespace std;
using namespace cv;
namespace DGFace{

Recognition::Recognition(void) {
}

Recognition::~Recognition(void) {
}

static void preprocess(const vector<Mat> &faces,  vector<Mat> &pro_faces) {
    Ptr<CLAHE> clahe = createCLAHE();
    pro_faces.reserve(faces.size());
    Mat pro_face;
    vector<Mat> channels;
    for (auto face : faces) {
        split(face, channels);
        clahe->setClipLimit(0.5);
        clahe->setTilesGridSize(Size(2,2));
        for (size_t i = 0; i < channels.size(); i++) {
            clahe->apply(channels[i], channels[i]);
        }
        merge(channels, pro_face);
        pro_faces.emplace_back(pro_face.clone());
    }
}

void Recognition::recog(const vector<Mat> &faces, vector<RecogResult> &results, const string &pre_process) {
    vector<Mat> pro_faces;
    if (pre_process == "CLAHE") {
        preprocess(faces, pro_faces);
        recog_impl(pro_faces, results);
    } else {
        recog_impl(faces, results);
    }
}
void Recognition::recog(const vector<Mat> &faces, const std::vector<AlignResult>& alignment, vector<RecogResult> &results, const string &pre_process) {
    vector<Mat> pro_faces;
    if (pre_process == "CLAHE") {
        preprocess(faces, pro_faces);
        recog_impl(pro_faces, alignment, results);
    } else {
        recog_impl(faces, alignment, results);
    }
}
}

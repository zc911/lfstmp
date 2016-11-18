#include <transformation.h>
#include "dgface_utils.h"

using namespace std;
using namespace cv;
namespace DGFace {

Transformation::Transformation() {
}

Transformation::~Transformation(void) {
}

void Transformation::transform(const Mat& img, const AlignResult& src_align,
								Mat& transformed_img, AlignResult& transformed_align) {
	transform_impl(img, src_align, transformed_img, transformed_align);	
}
}

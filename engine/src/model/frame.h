/*
 * Frame.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef FRAME_H_
#define FRAME_H_

#include <vector>
#include <pthread.h>

#include "payload.h"

namespace deepglint {

using namespace std;

typedef enum {

} FrameType;

typedef enum {

} FrameStatus;

class Frame {
public:
	Frame();
	virtual ~Frame();
protected:
	Identification id_;
	Timestamp timestamp_;
	//volatile FrameType type_;
	volatile FrameStatus status_;
	//pthread_mutex_t status_lock_;
	//pthread_mutex_t type_lock_;
	Payload *payload_;
	Operation operation_;
	// base pointer
	vector<Object *> objects_;
};

class RenderableFrame: public Frame {
	RenderableFrame();
	~RenderableFrame();
private:
	cv::Mat render_data_;
};

class FrameBatch {
public:
	FrameBatch();
	~FrameBatch();
private:
	Identification id_;
	unsigned int batch_size_;
	vector<Frame *> frames_;
};

// TODO
class RankData : public Frame{
//	Frame *frame_;
	vector<Feature> features_;
};

}

#endif /* FRAME_H_ */

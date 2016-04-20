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

using namespace std;

namespace dg
{

typedef enum
{

} FrameType;

enum FrameStatus
{
	FRAME_STATUS_INIT = 0,
	FRAME_STATUS_DETECTED = 1,
	FRAME_STATUS_FINISHED = 128
};

class Frame
{
public:
	Frame(const Identification id) :
			id_(id), timestamp_(0), status_(FRAME_STATUS_INIT), operation_(0), payload_(
					0)
	{
	}

	Frame(const Identification id, unsigned int width, unsigned int height,
			unsigned char *data) :
			id_(id), timestamp_(0), status_(FRAME_STATUS_INIT), operation_(0)
	{
		payload_ = new Payload(id_, width, height, data);
	}
	Frame(const Identification id, Mat img) :
			id_(id), timestamp_(0), status_(FRAME_STATUS_INIT), operation_(0)
	{
		payload_ = new Payload(id_, img);
	}
	virtual ~Frame()
	{
		if (payload_)
			delete payload_;
	}

	Identification id() const
	{
		return id_;
	}

	void set_id(Identification id)
	{
		id_ = id;
	}

	const vector<Object*>& objects() const
	{
		return objects_;
	}

	void put_object(Object *obj)
	{
		for (vector<Object *>::iterator itr = objects_.begin();
				itr != objects_.end(); ++itr)
		{
			Object *old_obj = *itr;

			if (old_obj->id() == obj->id())
			{
				delete old_obj;
				itr = objects_.erase(itr);
				break;
			}
		}
		objects_.push_back(obj);
	}

	Object* get_object(Identification id)
	{
		for (vector<Object *>::iterator itr = objects_.begin();
				itr != objects_.end(); ++itr)
		{
			Object *obj = *itr;
			if (obj->id() == id)
			{
				return *itr;
			}
		}
		return NULL;
	}

	void set_objects(const vector<Object*>& objects)
	{
		objects_ = objects;
	}

	Operation operation() const
	{
		return operation_;
	}

	void set_operation(Operation operation)
	{
		operation_ = operation;
	}

	Payload* payload() const
	{
		return payload_;
	}

	void set_payload(Payload* payload)
	{
		payload_ = payload;
	}

	volatile FrameStatus status() const
	{
		return status_;
	}

	void set_status(volatile FrameStatus status)
	{
		status_ = status;
	}

	Timestamp timestamp() const
	{
		return timestamp_;
	}

	void set_timestamp(Timestamp timestamp)
	{
		timestamp_ = timestamp;
	}

	int get_object_size()
	{
		return objects_.size();
	}

protected:
	Identification id_;
	Timestamp timestamp_;
	volatile FrameStatus status_;
	Operation operation_;
	Payload *payload_;
	// base pointer
	vector<Object *> objects_;
};

class RenderableFrame : public Frame
{
public:
	RenderableFrame();
	~RenderableFrame();
private:
	cv::Mat render_data_;
};

// just derive the base class
class FrameBatch : private Frame
{
public:
	FrameBatch();
	~FrameBatch();
private:
	Identification id_;
	unsigned int batch_size_;
	vector<Frame *> frames_;
};

// TODO
class RankData : public Frame
{
	Frame *frame_;
	vector<Box> hotspots_;
	vector<Feature> features_;
};

}

#endif /* FRAME_H_ */

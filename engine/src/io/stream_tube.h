/*
 * stream_tube.h
 *
 *  Created on: Feb 17, 2016
 *      Author: chenzhen
 */

#ifndef STREAM_TUBE_H_
#define STREAM_TUBE_H_

#include <pthread.h>
#include <string>
#include "dgmedia.h"
#include "model/frame.h"
#include "model/ringbuffer.h"

using namespace std;
using namespace dgmedia;

namespace dg {

const static string SUPPORT_NETSTREAM = "rtsp";
const static string SUPPORT_FILE_MP4 = "mp4";
const static string SUPPORT_FILE_AVI = "avi";
const static char FILE_POSTFIX_SEPERATOR = '.';
const static char NET_PROTOCOL_SEPERATOR = ':';

/**
 * This class read video data from decoder and put into
 * the ring buffer.
 */
class StreamTube {
 public:
    /**
     * Constructor
     */
    StreamTube(RingBuffer *buffer, const string addr, const unsigned int fps,
               const unsigned int width, const unsigned int height,
               const bool repeat = true);
    ~StreamTube();

    /**
     * Start reading data. This method will be blocked
     */
    int Start();
    /**
     * Start reading data. This method reads data in asynchronous
     */
    int StartAsyn();

    /**
     * Stop data reading. Not implemented
     */
    int Stop();

    /**
     * Check video address. Not implemented
     */
    bool CheckAddr();

    /**
     * The data read and put method
     */
    void *read();

    /**
     * Frame decode finished call-back function
     */
    static void frameDecodeFinished(unsigned char *data, int size,
                                    Frameset info);
    /**
     * Video end call-back function
     */
    static void eosReadched();
    static void onRuntimeErrorReached(Error error);

 private:
    int initDecoder();
 private:
    static RingBuffer *buffer_;
    static unsigned long long frame_id_;
    string stream_addr_;
    unsigned int width_;
    unsigned int height_;
    unsigned int fps_;

    bool repeat_;
    PersianPipeline *decoder_;
    pthread_t tid_;
};
}
#endif /* STREAM_TUBE_H_ */

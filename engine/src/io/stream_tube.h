/*
 * stream_tube.h
 *
 *  Created on: Feb 17, 2016
 *      Author: chenzhen
 */

#ifndef STREAM_TUBE_H_
#define STREAM_TUBE_H_

#include <string>
#include <pthread.h>
#include "dgmedia.h"
#include "ringbuffer.h"

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
    StreamTube(RingBuffer *buffer_, const string addr, const unsigned int fps,
               const unsigned int width, const unsigned int height, int decoder_latency, string decoder_protocol,
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
    static unsigned long long data_id_;

    string stream_addr_;
    unsigned int max_width_;
    unsigned int max_height_;
    unsigned int fps_;
    string decode_latency_ns_;
    string decoder_protocol_hex_;
    bool repeat_;
    PersianPipeline *decoder_;
    pthread_t tid_;
};

}
#endif /* STREAM_TUBE_H_ */

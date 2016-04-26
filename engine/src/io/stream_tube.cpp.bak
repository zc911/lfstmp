#include "stream_tube.h"
#include "glog/logging.h"
#include "string_util.h"

namespace dg {

RingBuffer* StreamTube::buffer_ = NULL;
unsigned long long StreamTube::frame_id_ = 0;

StreamTube::StreamTube(RingBuffer *buffer, const string addr,
                       const unsigned int fps, const unsigned int width,
                       const unsigned int height, const bool repeat)
        : stream_addr_(addr),
          fps_(fps),
          width_(width),
          height_(height),
          repeat_(repeat),
          tid_(NULL) {

    initDecoder();
    buffer_ = buffer;
    frame_id_ = 0;

}
StreamTube::~StreamTube() {
    decoder_->Stop();
    delete decoder_;
}

bool StreamTube::CheckAddr() {
// TODO
// check the stream_addr_ type and validation
    return true;
}
int StreamTube::Start() {
    CheckAddr();
    read();
    return 0;
}

int StreamTube::StartAsyn() {
    typedef void* (*FUNC)(void*);
    FUNC callback = (FUNC) &StreamTube::read;
    pthread_create(&tid_, NULL, callback, (void*) this);
    return 0;
}

int StreamTube::Stop() {
    decoder_->Stop();
    return 0;
}

void StreamTube::frameDecodeFinished(unsigned char *data, int size,
                                     Frameset info) {

    frame_id_++;
    DLOG(INFO)<< "Read Data: " << frame_id_ <<"-"<< info.src_width << "-"<<info.src_height << endl;
#ifdef __x86_64__
    buffer_->TryPut(info.src_width, info.src_height, data);
#else
    buffer_->TryPut(info.resize_width, info.resize_height, data);
#endif
}

void StreamTube::eosReadched() {
    // TODO
    DLOG(INFO)<< "EOS reached" << endl;
}
void StreamTube::onRuntimeErrorReached(Error error) {
    // TODO
    DLOG(INFO)<< "Run time error: " << error.err << endl;
}

void *StreamTube::read() {

    decoder_->Run();
    return NULL;
}

int StreamTube::initDecoder() {
    string prefix = findPrefix(stream_addr_, NET_PROTOCOL_SEPERATOR);
    string postfix = findPostfix(stream_addr_, FILE_POSTFIX_SEPERATOR);
    bool netStream = false;

    if (prefix == SUPPORT_NETSTREAM) {
        netStream = true;
    } else {
        string fileFormat;

        if (postfix == SUPPORT_FILE_AVI || postfix == SUPPORT_FILE_MP4) {
            fileFormat = postfix;
        } else {
            LOG(FATAL)<< "Not support file format: " << postfix;
            return -1;
        }
    }

    PixFmtType decoder_pftype = BGRA;

#ifdef __x86_64__
    if (netStream) {
        decoder_ = new RTSPFFDecH264Pipeline(stream_addr_.c_str(),
                                             decoder_pftype,
                                             StreamTube::frameDecodeFinished);

    } else {
        if (postfix == SUPPORT_FILE_AVI) {
            decoder_ = new AVIFFDecH264Pipeline(
                    stream_addr_.c_str(), decoder_pftype,
                    StreamTube::frameDecodeFinished);
        } else {
            decoder_ = new Mpeg4FFDecH264Pipeline(
                    stream_addr_.c_str(), decoder_pftype,
                    StreamTube::frameDecodeFinished);
        }
    }

#else
    if (netStream) {
        decoder_ = new RTSPOMXDecH264Pipeline(stream_addr_.c_str(),
                decoder_pftype,
                StreamTube::frameDecodeFinished);

    } else {
        if (postfix == SUPPORT_FILE_AVI) {
            decoder_ = new AVIOMXDecH264Pipeline(
                    stream_addr_.c_str(), decoder_pftype,
                    StreamTube::frameDecodeFinished);
        } else {
            decoder_ = new Mpeg4OMXDecH264Pipeline(
                    stream_addr_.c_str(), decoder_pftype,
                    StreamTube::frameDecodeFinished);
        }
    }
    decoder_->SetResizeResolution(width_, height_);

#endif

    decoder_->SetRepeat(repeat_);
    decoder_->RegisteRuntimeErrorReached(StreamTube::onRuntimeErrorReached);
    decoder_->RegisteEOSReached(StreamTube::eosReadched);
    Error error = decoder_->Initialize();
    if (error.code < 0) {
        LOG(FATAL)<< "Init decoder failed: " << error.err << endl;
        return -1;
    }
    return 0;
}
}
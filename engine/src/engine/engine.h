/*
 * engine.h
 *
 *  Created on: Feb 16, 2016
 *      Author: chenzhen
 */

#ifndef ENGINE_H_
#define ENGINE_H_
#include <pthread.h>

namespace deepglint {
template<typename TYPE, void (TYPE::*Process)()>
void* _start_thread_t(void *param) {
    TYPE *This = (TYPE*) param;
    This->Process();
    pthread_exit (NULL);
    return NULL;
}

/**
 * The basic engine interface.
 * An engine reads Frame data from ring buffer.
 * An engine aggregates several processors and invoke them to handle the frame
 * using some schedule method
 */
class Engine {
 public:
    Engine(RingBuffer *buffer)
            : tid_(NULL) {
    }

    virtual ~Engine() {
        //TODO release resources
    }
    virtual int Start() {
        Process();
        return 0;
    }

    /**
     * Start the engine processing in thread by invoking the process method.
     * The child class must implement the process method.
     */
    virtual int StartAsyn() {
        pthread_create(&tid_, NULL, _start_thread_t<Engine, &Engine::Process>,
                       (void*) this);
        return 0;
    }
    virtual int Stop() = 0;
    virtual int Release() = 0;

    void SetDisplay(bool display) {
        display_ = display;
    }
    bool IsDisplay() {
        return display_;
    }
    virtual void Process() = 0;

 protected:
    pthread_t tid_;

};

class ConfigableEngine : public Engine{

};
}
#endif /* ENGINE_H_ */

/*
 * engine.h
 *
 *  Created on: Feb 16, 2016
 *      Author: chenzhen
 */

#ifndef AUTO_ENGINE_H_
#define AUTO_ENGINE_H_
#include <pthread.h>

namespace dg {

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
class AutoEngine {
 public:
    AutoEngine()
            : tid_(NULL) {
    }

    virtual ~AutoEngine() {
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
        pthread_create(&tid_, NULL,
                       _start_thread_t<AutoEngine, &AutoEngine::Process>,
                       (void*) this);
        return 0;
    }
    virtual int Stop() = 0;
    virtual int Release() = 0;

    virtual void Process() = 0;

 protected:
    pthread_t tid_;

};

}
#endif /* ENGINE_H_ */

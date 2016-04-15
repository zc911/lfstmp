/*
 * display.cpp
 *
 *  Created on: Jan 5, 2016
 *      Author: haoquan
 */
#include <unistd.h>
#include <string>
#include <glog/logging.h>
#include "display.h"

using namespace cv;

namespace dg {

Displayer * Displayer::self_ = NULL;

void Displayer::glutDisplayIt() {
    Displayer::self_->displayFrame();
}

void Displayer::glutTimerFuncIt(int n) {
    Displayer::self_->timeFunc(0);
}

void Displayer::glutIdleIt() {
    glutPostRedisplay();
}

Displayer::Displayer(RingBuffer* buffer, const string winName, int width,
                     int height, int snapWidth, int snapHeight, const int fps) {
    win_name_ = winName;
    width_ = width + snapWidth;
    height_ = height;
    fps_ = fps;
    frame_iterval_ = 1000 / fps;
    int argc = 1;
    char *argv[1];
    argv[0] = "VSD";
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width_, height_);
    glutCreateWindow(winName.c_str());
    glEnable(GL_TEXTURE_2D);

    glutDisplayFunc(glutDisplayIt);
    glutTimerFunc(frame_iterval_, glutTimerFuncIt, 0);

    ring_buffer_ = buffer;
    display_pointer_ = 0;

    Displayer::self_ = this;

}

void Displayer::displayFrame() {

    glRasterPos3f(-1.0f, 1.0f, 0);
    glPixelZoom(1.0f, -1.0f);
    Frame *f = ring_buffer_->Get(display_pointer_);
    if (f == NULL) {
        return;
    }
    if (f->status() != FRAME_STATUS_DETECTED) {
        return;
    }
    display_pointer_++;
    glDrawPixels(width_, height_, GL_BGRA, GL_UNSIGNED_BYTE,
                 f->payload()->data().data);

    glutSwapBuffers();
    f->set_status(FRAME_STATUS_FINISHED);

//
//    if (f == NULL) {
//        return;
//    }
//
//    if ((f->GetStatus() & FRAME_STATUS_PROCESSED) == 0
//            || (f->GetStatus() & FRAME_STATUS_DISPLAYED) > 0) {
//
//        DLOG(INFO)<< ">>>>> Can not display: " << f << "-" << f->FrameId() << "-" << f->GetStatus() << endl;
//        usleep(frame_iterval_ * 1000);
//        return;
//    }
//
//    DLOG(INFO)<< ">>>>> Display frame: " << f <<"-" << f->FrameId() << "-" << f->GetStatus() << endl;
//

//
//    DLOG(INFO)<< ">>>>> End display: " << f << "-" << f->FrameId() << "-" << f->GetStatus() << endl;
//
//    t_profiler_str_ = "Display";
//    t_profiler_.update(t_profiler_str_);
//    if (profile_time_)
//        LOG(INFO)<< t_profiler_.getSmoothedTimeProfileString();
//
//    f->SetStatus(FRAME_STATUS_FINISHED, true);
//    DLOG(INFO)<< "Display frame: " << f << "-" << f->FrameId() << "-" << f->GetStatus() << endl;
//    display_pointer_++;
//    if (display_pointer_ >= buffer_size_)
//        display_pointer_ = 0;

}

void Displayer::timeFunc(int n) {
    displayFrame();
    glutTimerFunc(frame_iterval_, glutTimerFuncIt, 0);
}

void Displayer::Run() {
    glutMainLoop();
}

}

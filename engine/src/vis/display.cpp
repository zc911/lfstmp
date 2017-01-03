/*
 * display.cpp
 *
 *  Created on: Jan 5, 2016
 *      Author: haoquan
 */
#include <string>
#include <glog/logging.h>
#include "display.h"

using namespace cv;

namespace dg {

Displayer *Displayer::self_ = NULL;

void Displayer::glutDisplayIt() {
    Displayer::self_->displayFrame();
}

void Displayer::glutTimerFuncIt(int n) {
    Displayer::self_->timeFunc(0);
}

void Displayer::glutIdleIt() {
    glutPostRedisplay();
}

Displayer::Displayer(RingBuffer *buffer, const string winName, int width,
                     int height, int snapWidth, int snapHeight, const int fps) {
    win_name_ = winName;
    width_ = width;
    snap_width_ = snapWidth;
    snap_height_ = snapHeight;
    height_ = height;
    fps_ = fps;
    frame_iterval_ = 1000 / (fps / 2);
    int argc = 1;
    char *argv[1];
    argv[0] = "";
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width_ + snap_width_, height_);
    glutCreateWindow(winName.c_str());
    glEnable(GL_TEXTURE_2D);

    glutDisplayFunc(glutDisplayIt);
    glutTimerFunc(frame_iterval_, glutTimerFuncIt, 0);

    ring_buffer_ = buffer;
    display_pointer_ = 0;

    Displayer::self_ = this;

}

void Displayer::render(Frame *frame) {
    vector<Object *> &objs = frame->objects();
    Mat rdata = frame->payload()->data();
    for (auto o : objs) {
        const Detection &d = o->detection();
        cv::rectangle(rdata, d.box(), cv::Scalar(255, 0, 0));
    }
}


void Displayer::displayFrame() {

    glRasterPos3f(-1.0f, 1.0f, 0);
    glPixelZoom(1.0f, -1.0f);
    Frame *f = ring_buffer_->GetFrame(display_pointer_);
    if (f == NULL) {
        DLOG(INFO) << "Frame is NULL" << endl;
        return;
    }
    if (!f->CheckStatus(FRAME_STATUS_ABLE_TO_DISPLAY) || f->CheckStatus(FRAME_STATUS_FINISHED)) {
        DLOG(INFO) << "Can not display frame " << f->id() << ", status: " << f->status() << endl;
        return;
    }
    display_pointer_++;
    Mat data = f->payload()->data();


    if (data.rows == 0 || data.cols == 0) {
        cout << "Frame data is empty: " << f->id() << endl;
        return;
    }
    Mat displayData;
    render(f);


    if (data.cols != width_ || data.rows != height_) {
        cv::resize(data, displayData, cv::Size(width_, height_));
    }
    glDrawPixels(width_, height_, GL_BGRA, GL_UNSIGNED_BYTE,
                 displayData.data);

    glutSwapBuffers();
    f->set_status(FRAME_STATUS_FINISHED, false);

}

void Displayer::timeFunc(int n) {
    displayFrame();
    glutTimerFunc(frame_iterval_, glutTimerFuncIt, 0);
}

void Displayer::Run() {
    glutMainLoop();
}

}
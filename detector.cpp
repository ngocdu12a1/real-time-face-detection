//
// Created by du on 05/10/2020.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "detector.h"
#include <facedetectcnn.h>


#define DETECT_BUFFER_SIZE 0x20000

using namespace std;
using namespace cv;

Detector::Detector() {

}

Detector::Detector(int width, int height) {
    this->isFaceFound = false;
    this->width = width;
    this->height = height;
    this->pBuffer = (unsigned char*) malloc(DETECT_BUFFER_SIZE);
}

Detector::~Detector() {
    free(pBuffer);
}

bool Detector::detectFaceCNN(Mat image, vector<FaceRectangle> &result) {

    if(image.empty()){
        fprintf(stderr, "Image is NULL\n");
        return false;
    }

    int *pResults = NULL;


    if(!pBuffer){
        fprintf(stderr, "Cannot alloc buffer\n");
        return false;
    }

    pResults = facedetect_cnn(pBuffer, (unsigned char*) image.ptr(0), image.cols, image.rows, (int) image.step);

    if(*pResults == 0){
        fprintf(stdout, "No face detected\n");
        return false;
    }

    short* p = ((short*) (pResults + 1));

    FaceRectangle rect;
    rect.confidence = p[0];
    rect.x = p[1];
    rect.y = p[2];
    rect.w = p[3];
    rect.h = p[4];

    result.push_back(rect);

    return  true;
}

bool Detector::detectFace(Mat image, vector<FaceRectangle> &result) {
    bool res = false;

    if(isFaceFound){
        res = detectFaceAroundROI(image, result);
    }
    else {
        res = detectFaceInWholeImage(image, result);
    }

    if(res){
        isFaceFound = true;
        // set region of interest
        setRegionOfInterest(result[0]);
    }
    else {
        isFaceFound = false;
        // switch template matching (may be)
    }

    result.push_back(regionOfInterest);
    return res;
}

bool Detector::detectFaceAroundROI(Mat image, vector<FaceRectangle> &result) {

    Rect roi;
    roi.x = regionOfInterest.x;
    roi.y = regionOfInterest.y;
    roi.height = regionOfInterest.h;
    roi.width = regionOfInterest.w;

    Mat subImage = image(roi);

    bool res = detectFaceCNN(subImage, result);

    result[0].x += regionOfInterest.x;
    result[0].y += regionOfInterest.y;

    subImage.release();

    return res;
}

bool Detector::detectFaceInWholeImage(Mat image, vector<FaceRectangle> &result) {

    bool res = detectFaceCNN(image, result);

    if(res){
        isFaceFound = true;
    }
    else {
        isFaceFound = false;
    }

    return res;
}

void Detector::setRegionOfInterest(FaceRectangle face) {
    regionOfInterest.x = (face.x - (int) (0.3*face.w)) > 0? (face.x - (int) (0.3*face.w)):0;
    regionOfInterest.y = (face.y - (int) (0.3*face.h)) > 0? (face.y - (int) (0.3*face.h)):0;

    int x1 = face.x + (int) (1.3*face.w) > width ? width:face.x + (int) (1.3*face.w);
    int y1 = face.y + (int) (1.3*face.h) > height ? height:face.y + (int) (1.3*face.h);

    regionOfInterest.w = x1 - regionOfInterest.x;
    regionOfInterest.h = y1 - regionOfInterest.y;
}




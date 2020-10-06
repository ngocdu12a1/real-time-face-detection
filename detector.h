//
// Created by du on 05/10/2020.
//

#ifndef HELLODLIB_DETECTOR_H
#define HELLODLIB_DETECTOR_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <facedetectcnn.h>

#define DETECT_BUFFER_SIZE 0x20000

using namespace std;
using namespace cv;

struct FaceRectangle{
    int x, y, w, h, confidence;
};

class Detector{

public:

    Detector();
    Detector(int width, int height);
    ~Detector();
    bool detectFace(Mat image, vector<FaceRectangle> &result);
    vector<FaceRectangle> detectFaceDLibHOG(Mat image);
    vector<FaceRectangle> detectFaceOpenCV(Mat image);
    bool detectFaceCNN(Mat image, vector<FaceRectangle> &result);

private:
    bool detectFaceAroundROI(Mat image, vector<FaceRectangle> &result);
    bool detectFaceInWholeImage(Mat image, vector<FaceRectangle> &result);

    void setRegionOfInterest(FaceRectangle face);

    bool isFaceFound;
    FaceRectangle regionOfInterest;
    int width, height;
    unsigned char* pBuffer;
};
#endif //HELLODLIB_DETECTOR_H

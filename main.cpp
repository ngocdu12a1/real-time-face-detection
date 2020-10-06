#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#include "detector.h"
#include <sys/time.h>

using namespace std;
using namespace cv;


int main() {
    cv::VideoCapture cap("/home/du/Downloads/data720.mp4");
    cv::Mat temp;
    vector<FaceRectangle> result;
    Detector* detector = new Detector(cap.get(CAP_PROP_FRAME_WIDTH),
                                        cap.get(CAP_PROP_FRAME_HEIGHT));
    char score[256];
    int count = 0;

    struct timeval start, stop;

    if (!cap.isOpened()) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }


    // Grab and process frames until the main window is closed by the user.
    while (true) {
        // Grab a frame

        if (!cap.read(temp)) {
            break;
        }




        gettimeofday(&start, NULL);
        bool res = detector->detectFace(temp, result);
        gettimeofday(&stop, NULL);

        long time = (stop.tv_sec - start.tv_sec) * 1000000  + (stop.tv_usec - start.tv_usec);
     //   fprintf(stdout, "FPS is: %f\n", 1000000/(float)time);

        if(res){
            FaceRectangle rect = result[0];
            FaceRectangle roi = result[1];

            int confidence = rect.confidence;
            int x = rect.x;
            int y = rect.y;
            int w = rect.w;
            int h = rect.h;

            snprintf(score, 256, "%d", confidence);
            putText(temp, score, Point(x, y - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            rectangle(temp, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
            rectangle(temp, Rect(roi.x, roi.y, roi.w, roi.h), Scalar(0, 0, 255), 2);
            cout <<count ++ << " " <<  x << " " << y << " " << w << " " << h << endl;
        }


        char name[256];
        sprintf(name, "%d.jpg", count);

        imshow("face", temp);
        char c=(char)waitKey(1);
        if(c==27)
        break;

        result.clear();
        memset(score, 0, 256);
    }

    delete (detector);
    free(score);
}




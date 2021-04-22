#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

std::vector<cv::KeyPoint> SIFTonFrame(cv::Mat frame) {

	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
	siftPtr->detect(frame, keypoints);

	return keypoints;
}


int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: multiFeatueDetection <video path>\n");
        return -1;
    }

	cv::VideoCapture cap(argv[1]);
 	if (cap.isOpened() == false)
	{
  		fprintf(stderr,"Cannot open the video file %s\n",argv[1]);
   		return -1;
  	}

	double fps = cap.get(cv::CAP_PROP_FPS);
	fprintf(stderr,"Video FPS is %f frames/s\n",fps);

	std::string windowName = "Output";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	while(true) {
		cv::Mat in_frame;
		cv::Mat out_frame;

		bool bSuccess = cap.read(in_frame);
		if(bSuccess == false) {
			fprintf(stderr,"EOF\n");
			break;
		}
		out_frame = in_frame.clone();

		//SIFT
	    std::vector<cv::KeyPoint> keypoints = SIFTonFrame(in_frame);
		cv::drawKeypoints(out_frame,keypoints,out_frame);

		//corner detection



		//output
		cv::imshow(windowName,out_frame);
		cv::waitKey(10); //delay TODO remove & use timer
	}
}


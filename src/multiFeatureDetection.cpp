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

		cv::Mat white_mask;
		cv::Mat gs_frame;

		white_mask.create(in_frame.size(),in_frame.type());
		cv::cvtColor(in_frame,gs_frame,cv::COLOR_BGR2GRAY);

		//SIFT
	    std::vector<cv::KeyPoint> keypoints = SIFTonFrame(in_frame);

		//corner detection
		cv::Mat harris_dst = cv::Mat::zeros(gs_frame.size(),CV_32FC1);
		cv::cornerHarris(gs_frame,harris_dst,3,3,0.04);

			//normalize
		cv::Mat dst_norm;
		cv::Mat dst_norm_scaled;
		cv::normalize(harris_dst, dst_norm, 0,255,cv::NORM_MINMAX, CV_32FC1,cv::Mat());
		cv::convertScaleAbs(dst_norm,dst_norm_scaled);


			//draw circle



		//canny

		cv::Mat detected_edges;
		white_mask = cv::Scalar::all(255);
		cv::blur(gs_frame,detected_edges,cv::Size(3,3));
		cv::Canny(detected_edges, detected_edges,50,150,3);


		//apply changes
		white_mask.copyTo(out_frame,detected_edges);
		cv::drawKeypoints(out_frame,keypoints,out_frame);

		//draw harris circles
		for(int i = 0; i < dst_norm.rows; i++) {
			for(int j = 0; j < dst_norm.cols; j++) {
				if((int)dst_norm.at<float>(i,j) > 200) {
					cv::circle(out_frame,cv::Point(j,i),30,cv::Scalar(0),8,8,0);
				}
			}
		}

		//output
		cv::imshow(windowName,out_frame);
		cv::waitKey(100); //delay TODO remove & use timer
	}
}


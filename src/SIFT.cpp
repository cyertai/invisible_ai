#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

int main(int argc, char** argv)
{
    //const cv::Mat input = cv::imread("input.jpg", 0); //Load as grayscale
    const cv::Mat input = cv::imread("/home/glenn/src/invisible_ai/data/lenna.png", 0); //Load as grayscale

    cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    siftPtr->detect(input, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::imwrite("sift_result.jpg", output);

    return 0;
}



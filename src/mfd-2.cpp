#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <pthread.h>
#include <thread>
#include <vector>
#include <unistd.h>
#include <math.h>

#define NO_TASK -1
#define SIFT_TASK 0
#define CANNY_TASK 1
#define HARRIS_TASK 2

#define CANNY_KERNEL 3
#define CANNY_THRESH1 50
#define CANNY_THRESH2 150
#define CANNY_BLUR_SIZE 3

#define HARRIS_BLOCK_SIZE 5
#define HARRIS_KERNEL 11
#define HARRIS_FREE 0.04
#define HARRIS_THRESH 150

#define SIFT_N_FEATURES 2000

#define SIFT_FPS 5
#define CANNY_FPS 25
#define HARRIS_FPS 5

#define HARRIS_CIRCLE_R 30
#define HARRIS_CIRCLE_THICKNESS 8

#define MAX_THREADS 3 //one for each task type

//thread for communication with worker threads;
typedef struct {
	int threadNum;
	bool selfDestruct;
	bool isJoinable;
	pthread_mutex_t mutex1;
} TPN_t;

struct taskOb {
	long frame_num;
	int task_id;
	cv::Mat input_image;

	~taskOb(){
	this->input_image.release();
	}
};

taskOb* newTaskOb(long frame_num,int task_id,cv::Mat input_image) {
	auto task = new taskOb;
	task->frame_num = frame_num;
	task->task_id = task_id;
	//task->input_image;
	input_image.copyTo(task->input_image);
	return task;
}

std::vector<std::thread> G_threads;

//Lists for naive task communication
std::list<int> G_taskList;
//std::list<taskOb*> G_taskList;
pthread_mutex_t G_taskListMutex;

std::list<cv::Mat> G_SiftTaskList;
std::list<std::vector<cv::KeyPoint>> G_SiftResponseList;

std::list<cv::Mat> G_CannyTaskList;
std::list<cv::Mat> G_CannyResponseList;

std::list<cv::Mat> G_HarrisTaskList;
std::list<std::vector<cv::Point>> G_HarrisResponseList;

std::list<cv::Mat> G_frameBuffer;

std::vector<cv::KeyPoint> SIFTonFrame(cv::Mat frame) {
	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
	siftPtr->detect(frame, keypoints);

	return keypoints;
}

cv::Mat cannyRoutine(cv::Mat gs_frame,cv::Mat white_mask) {
	cv::Mat detected_edges;
	white_mask = cv::Scalar::all(255);
	cv::blur(gs_frame,detected_edges,cv::Size(CANNY_BLUR_SIZE,CANNY_BLUR_SIZE));
	cv::Canny(detected_edges, detected_edges,CANNY_THRESH1,CANNY_THRESH2,
			  CANNY_KERNEL);
	return detected_edges;
}


std::vector<cv::Point> harrisCornerDetection(cv::Mat gs_frame) {
		std::vector<cv::Point> harrisResultsVector;
		cv::Mat harris_dst = cv::Mat::zeros(gs_frame.size(),CV_32FC1);
		cv::cornerHarris(gs_frame,harris_dst,HARRIS_BLOCK_SIZE,
						 HARRIS_KERNEL,HARRIS_FREE);
		cv::Mat dst_norm;
		cv::Mat dst_norm_scaled;
		cv::normalize(harris_dst, dst_norm, 0,255,cv::NORM_MINMAX,
					  CV_32FC1,cv::Mat());
		cv::convertScaleAbs(dst_norm,dst_norm_scaled);

		for(int i = 0; i < dst_norm.rows; i++) {
			for(int j = 0; j < dst_norm.cols; j++) {
				if((int)dst_norm.at<float>(i,j) > HARRIS_THRESH) {
					harrisResultsVector.push_back(cv::Point(j,i));
				}
			}
		}
		return harrisResultsVector;
}

cv::Mat applyFrameMarking(cv::Mat white_mask,
						  cv::Mat detected_edges,
						  int detected_edges_flag,
						  cv::Mat out_frame,
						  std::vector<cv::KeyPoint> keypoints,
						  std::vector<cv::Point> harrisResultsVector)
{
		//Canny
		if(detected_edges_flag) {
			white_mask.copyTo(out_frame,detected_edges);
		}
		//SIFT
		cv::drawKeypoints(out_frame,keypoints,out_frame);

		//Harris Corners
		for(int i = 0; i < harrisResultsVector.size(); i++)
		{
			//Circle size hard coded - no defines
			cv::circle(out_frame,harrisResultsVector[i],HARRIS_CIRCLE_R,
					   cv::Scalar(0),HARRIS_CIRCLE_THICKNESS,8,0);
		}
		harrisResultsVector.clear();
		return out_frame;
}


int initThreadNode(TPN_t* td,int threadNum) {
    td->threadNum = threadNum;
    td->selfDestruct = false;
    td->isJoinable = false;
	td->mutex1 = PTHREAD_MUTEX_INITIALIZER;
    return 0;
}

TPN_t** initThreadPoolData(int numThreads) {
    TPN_t** threadData = (TPN_t**)malloc(sizeof(TPN_t*)*numThreads);
    for(int i = 0; i < numThreads; i ++) {
        threadData[i] = (TPN_t*)malloc(sizeof(TPN_t));
    }
    return threadData;
}

int internalShutdownPool(TPN_t** threadData,int numThreads) {
    //recallWorkerThreads
    for(int i = 0; i < numThreads; i ++) {
            pthread_mutex_lock(&threadData[i]->mutex1);
            threadData[i]->isJoinable = true;
            pthread_mutex_unlock(&threadData[i]->mutex1);
    }

    for(int i = 0; i < numThreads; i ++) {
            G_threads[i].join();
    }

    //cleanup threadData
    for(int i = 0; i < numThreads; i ++) {
        free(threadData[i]);
    }
    free(threadData);

    return 0;
}

int doWorkerThreadWork(){
		/*
		int taskType = NO_TASK;
		int frame_num = -1;
		cv::Mat work;
		if(!G_taskList.empty()) {
			if(pthread_mutex_trylock(&G_taskListMutex)==0) {
				taskType = G_taskList.front()->task_id;
				frame_num = G_taskList.front()->frame_num;
				work = G_taskList.front()->input_image.clone();
				G_taskList.front()->input_image.release();
				delete G_taskList.front();
				G_taskList.pop_front();
				pthread_mutex_unlock(&G_taskListMutex);
			} else {
				pthread_mutex_unlock(&G_taskListMutex);

			}
		}

		if(taskType == CANNY_TASK) {
			cv::Mat gs_frame;
			fprintf(stderr,"CANNY\n");
			cv::Mat white_mask;
			cv::cvtColor(work,gs_frame,cv::COLOR_BGR2GRAY);
			white_mask.create(work.size(),work.type());
			cv::Mat detected_edges = cannyRoutine(gs_frame,white_mask);
			pthread_mutex_lock(&G_taskListMutex);
			G_CannyResponseList.push_back(detected_edges);
			pthread_mutex_unlock(&G_taskListMutex);
		}

		if(taskType == SIFT_TASK) {
			fprintf(stderr,"SIFT\n");
	    	std::vector<cv::KeyPoint> keypoints = SIFTonFrame(work);

			pthread_mutex_lock(&G_taskListMutex);
			G_SiftResponseList.push_back(keypoints);
			pthread_mutex_unlock(&G_taskListMutex);

		}

		if(taskType == HARRIS_TASK) {
			fprintf(stderr,"HARRIS\n");
			cv::Mat gs_frame;
			cv::cvtColor(work,gs_frame,cv::COLOR_BGR2GRAY);

			std::vector<cv::Point> harrisResultsVector =
			harrisCornerDetection(gs_frame);

			pthread_mutex_lock(&G_taskListMutex);
			G_HarrisResponseList.push_back(harrisResultsVector);
			pthread_mutex_unlock(&G_taskListMutex);
		}*/

		if(!G_CannyTaskList.empty()) {
			if(pthread_mutex_trylock(&G_taskListMutex)==0) {
				cv::Mat work = G_CannyTaskList.front().clone();
				G_CannyTaskList.front().release();
				G_CannyTaskList.pop_front();

				pthread_mutex_unlock(&G_taskListMutex);

				cv::Mat gs_frame;
				cv::Mat white_mask;

				cv::cvtColor(work,gs_frame,cv::COLOR_BGR2GRAY);
				white_mask.create(work.size(),work.type());
				cv::Mat detected_edges = cannyRoutine(gs_frame,white_mask);

				pthread_mutex_lock(&G_taskListMutex);
				G_CannyResponseList.push_back(detected_edges);
				pthread_mutex_unlock(&G_taskListMutex);
			}
		}

		if(!G_SiftTaskList.empty()) {
			if(pthread_mutex_trylock(&G_taskListMutex)==0) {
				cv::Mat work = G_SiftTaskList.front().clone();
				G_SiftTaskList.front().release();
				G_SiftTaskList.pop_front();
				pthread_mutex_unlock(&G_taskListMutex);

	    		std::vector<cv::KeyPoint> keypoints = SIFTonFrame(work);

				pthread_mutex_lock(&G_taskListMutex);
				G_SiftResponseList.push_back(keypoints);
				pthread_mutex_unlock(&G_taskListMutex);
			}
		}

		if(!G_HarrisTaskList.empty()) {
			if(pthread_mutex_trylock(&G_taskListMutex)==0) {
				cv::Mat work = G_HarrisTaskList.front().clone();
				G_HarrisTaskList.front().release();
				G_HarrisTaskList.pop_front();
				pthread_mutex_unlock(&G_taskListMutex);

				cv::Mat gs_frame;
				cv::cvtColor(work,gs_frame,cv::COLOR_BGR2GRAY);

				std::vector<cv::Point> harrisResultsVector =
					harrisCornerDetection(gs_frame);

				pthread_mutex_lock(&G_taskListMutex);
				G_HarrisResponseList.push_back(harrisResultsVector);
				pthread_mutex_unlock(&G_taskListMutex);
			}
		}
	return 0;
}

 int workerThreadLoop(TPN_t* tn) {
	while(1) {
 	   //check if the thread has been asked to join
        if(pthread_mutex_trylock(&tn->mutex1)==0){
            if(tn->isJoinable == true) {
                pthread_mutex_unlock(&tn->mutex1);
                return 0;
            } else {
                pthread_mutex_unlock(&tn->mutex1);
            }
        }
        //check for work to do from the queue
		doWorkerThreadWork();
		//encourage scheduling of other threads
        usleep(1);
    }
    //should never reach
    return -1;
}

int displayOutput(int* frameCount, double fps){
	std::string windowName = "Frame";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::Mat base_frame;

	if(!G_frameBuffer.empty() &&
	  (!G_SiftResponseList.empty()) &&
	  (!G_CannyResponseList.empty()) &&
      (!G_HarrisResponseList.empty())) {
		fprintf(stderr,"HERE!!\n");
		pthread_mutex_lock(&G_taskListMutex);

		base_frame = G_frameBuffer.front().clone();
		G_frameBuffer.pop_front();

		pthread_mutex_unlock(&G_taskListMutex);

	} else {
		return 0;
	}

	int SiftInt = !(*frameCount%(int(fps/5)));
	int CannyInt = !(*frameCount%(int(fps/25)));
	int HarrisInt = !(*frameCount%(int(fps/5)));
	int taskCount =  SiftInt + CannyInt + HarrisInt;

	frameCount++;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Point> harrisResultsVector;
	cv::Mat detected_edges;

	cv::Mat white_mask;
	white_mask.create(base_frame.size(),base_frame.type());
	white_mask = cv::Scalar::all(255);
	int haveCannyData = 0;

	while(taskCount > 0) {
		if(SiftInt) {
			if(pthread_mutex_trylock(&G_taskListMutex) == 0) {
				if((!G_SiftResponseList.empty()) && (taskCount > 0)) {
					keypoints =	G_SiftResponseList.front();
					G_SiftResponseList.pop_front();
					pthread_mutex_unlock(&G_taskListMutex);
					taskCount--;
					SiftInt--;
				} else {
					pthread_mutex_unlock(&G_taskListMutex);
				}
			}
		}
		if(CannyInt) {
			if(pthread_mutex_trylock(&G_taskListMutex) == 0) {
				if(!G_CannyResponseList.empty() && taskCount > 0) {
					detected_edges = G_CannyResponseList.front().clone();
					G_CannyResponseList.front().release();
					G_CannyResponseList.pop_front();
					pthread_mutex_unlock(&G_taskListMutex);
					haveCannyData = 1;
					taskCount--;
					CannyInt--;
				} else {
					pthread_mutex_unlock(&G_taskListMutex);
				}
			}
		}
		if(HarrisInt) {
			if(pthread_mutex_trylock(&G_taskListMutex) == 0) {
				if(!G_HarrisResponseList.empty() && taskCount > 0) {
					harrisResultsVector = G_HarrisResponseList.front();
					G_HarrisResponseList.pop_front();
					pthread_mutex_unlock(&G_taskListMutex);
					taskCount--;
					HarrisInt--;
				} else {
					pthread_mutex_unlock(&G_taskListMutex);
				}
			}
		}
		usleep(1);
		}


		applyFrameMarking(white_mask,detected_edges,base_frame,keypoints,
						  harrisResultsVector,haveCannyData);

		//output
		//cv::imshow(windowName,out_frame);
		cv::imshow("Frame",base_frame);
		cv::waitKey(10000); //delay TODO remove & use timer

	return 0;
}

int displayThreadLoop(TPN_t* tn, double fps) {
	//init variables
	int* frameCount = (int*)malloc(sizeof(int));
	*frameCount = 0;

	while(1) {
 	   //check if the thread has been asked to join
        if(pthread_mutex_trylock(&tn->mutex1)==0){
            if(tn->isJoinable == true) {
                pthread_mutex_unlock(&tn->mutex1);
				free(frameCount);
                return 0;
            } else {
                pthread_mutex_unlock(&tn->mutex1);
            }
        }
        //check for work to do from the queue

		displayOutput(frameCount,fps);

		//encourage scheduling of other threads
        usleep(1);
    }
	free(frameCount);
    //should never reach
    return -1;
}

int processVideo(cv::VideoCapture cap, double fps) {
	long frameCount = 0;
	while(true) {
		cv::Mat in_frame;
		if(!cap.read(in_frame)) {
			fprintf(stderr,"EOF\n");
			break;
		}
		frameCount++;

		//send frames out to worker queue for each applicable type
		int taskCount = 0;
		pthread_mutex_lock(&G_taskListMutex);

		if(frameCount%(int(fps/SIFT_FPS))==0) {
			G_SiftTaskList.push_back(in_frame);
			G_taskList.push_back(SIFT_TASK);
			taskCount ++;
		}

		if(frameCount%(int(fps/CANNY_FPS))==0) {
			G_CannyTaskList.push_back(in_frame);
			G_taskList.push_back(CANNY_TASK);
			detected_edges_flag = 1;
			taskCount ++;
		}


		if(frameCount%(int(fps/HARRIS_FPS))==0) {
			G_HarrisTaskList.push_back(in_frame);
			G_taskList.push_back(HARRIS_TASK);
			taskCount ++;
		}

/*
		if(frameCount%(int(fps/5))==0) {
			auto task = newTaskOb(frameCount,SIFT_TASK,in_frame);
			G_taskList.push_back(task);
			taskCount ++;
		}

		if(frameCount%(int(fps/25))==0) {
			auto task = newTaskOb(frameCount,CANNY_TASK,in_frame);
			G_taskList.push_back(task);
			taskCount ++;
		}


		if(frameCount%(int(fps/5))==0) {
			auto task = newTaskOb(frameCount,HARRIS_TASK,in_frame);
			G_taskList.push_back(task);
			taskCount ++;
		}
*/
		G_frameBuffer.push_back(in_frame);
		pthread_mutex_unlock(&G_taskListMutex);



	}

	while((!G_frameBuffer.empty()))
	{
		usleep(1000);
	}
	return 0;
}




int main(int argc, char** argv )
{
	//begin ARG parsing
    if ( argc != 2 )
    {
        printf("usage: ./mfd-2 <video path>\n");
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

	//End ARG parsing
	//setup threadpool


	int hardwareThreads = std::thread::hardware_concurrency();
	fprintf(stderr,"threads %i\n\n",hardwareThreads);
	TPN_t** threadData = initThreadPoolData(hardwareThreads+1); //+1 for disp thrd

	int i;
    for(i = 0; i < hardwareThreads; i++) {
        initThreadNode(threadData[i],i);
        G_threads.emplace_back(workerThreadLoop,threadData[i]);
    }

	initThreadNode(threadData[i],i); //init display thread
	G_threads.emplace_back(displayThreadLoop,threadData[i],fps);

	G_taskListMutex = PTHREAD_MUTEX_INITIALIZER;

	processVideo(cap,fps); //main processing loop

	internalShutdownPool(threadData,hardwareThreads+1); //join with threads
	return 0;
}


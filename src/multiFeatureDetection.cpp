#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <pthread.h>
#include <thread>
#include <vector>
#include <unistd.h>

#define SIFT_TASK 0
#define CANNY_TASK 1
#define HARRIS_TASK 2




//thread for communication with worker threads;
typedef struct {
	int threadNum;
	int taskType; //0 - SIFT, 1 - Edge, 2 - Corner; -1 = no task
	bool selfDestruct;
	bool isJoinable;
	pthread_mutex_t mutex1;
} TPN_t;

typedef struct {
	int taskType; //0 - SIFT, 1 - Edge, 2 - Corner; -1 = no task
	int frameCount;
	std::vector<cv::Mat> frameBuffer;
	//cv::Mat in_frame;
	//cv::Mat return_frame;
	pthread_mutex_t mutex1;
}taskOb_t;

std::vector<std::thread> G_threads;
std::list<taskOb_t*> G_taskList;
pthread_mutex_t G_listMutex;

std::list<taskOb_t*> G_workProduct;
pthread_mutex_t G_workProductMutex;


std::vector<cv::KeyPoint> SIFTonFrame(cv::Mat frame) {
	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
	siftPtr->detect(frame, keypoints);

	return keypoints;
}

cv::Mat cannyRoutine(cv::Mat gs_frame,cv::Mat white_mask) {
	cv::Mat detected_edges;
	white_mask = cv::Scalar::all(255);
	cv::blur(gs_frame,detected_edges,cv::Size(3,3));
	cv::Canny(detected_edges, detected_edges,50,150,3); //TODO put all constants in defs
	return detected_edges;
}

std::vector<cv::Point> harrisCornerDetection(cv::Mat gs_frame) {
		std::vector<cv::Point> harrisResultsVector;
		cv::Mat harris_dst = cv::Mat::zeros(gs_frame.size(),CV_32FC1);
		cv::cornerHarris(gs_frame,harris_dst,3,3,0.04);
		cv::Mat dst_norm;
		cv::Mat dst_norm_scaled;
		cv::normalize(harris_dst, dst_norm, 0,255,cv::NORM_MINMAX, CV_32FC1,cv::Mat());
		cv::convertScaleAbs(dst_norm,dst_norm_scaled);

		for(int i = 0; i < dst_norm.rows; i++) {
			for(int j = 0; j < dst_norm.cols; j++) {
				if((int)dst_norm.at<float>(i,j) > 200) {
					harrisResultsVector.push_back(cv::Point(j,i));
				}
			}
		}
		return harrisResultsVector;
}

cv::Mat applyFrameMarking(cv::Mat white_mask,
						  cv::Mat detected_edges,
						  cv::Mat out_frame,
						  std::vector<cv::KeyPoint> keypoints,
						  std::vector<cv::Point> harrisResultsVector)
{
		//Canny
		white_mask.copyTo(out_frame,detected_edges);

		//SIFT
		cv::drawKeypoints(out_frame,keypoints,out_frame);

		//Harris Corners
		for(int i = 0; i < harrisResultsVector.size(); i++)
		{
			cv::circle(out_frame,harrisResultsVector[i],30,cv::Scalar(0),8,8,0);
		}
		harrisResultsVector.clear();
		return out_frame;
}

taskOb_t* newTaskOb(int taskType,int frameCount,cv::Mat in_frame) {
	fprintf(stderr,"%i\n",__LINE__);
	//taskOb_t* item = (taskOb_t*)malloc(sizeof(taskOb_t));
	taskOb_t* item = new taskOb_t;
	fprintf(stderr,"%i\n",__LINE__);
    item->taskType = taskType;
	fprintf(stderr,"%i\n",__LINE__);
	item->frameCount = frameCount;
	fprintf(stderr,"%i\n",__LINE__);

	item->frameBuffer[0] = cv::Mat::zeros(in_frame.size(),in_frame.type());
	fprintf(stderr,"%i\n",__LINE__);
	if (taskType == SIFT_TASK ) {
		//item->in_frame = in_frame.clone();
	fprintf(stderr,"%i\n",__LINE__);
	} else {
	fprintf(stderr,"%i\n",__LINE__);
		cv::Mat gs_frame;
		gs_frame.create(in_frame.size(),in_frame.type());
		cv::cvtColor(in_frame,gs_frame,cv::COLOR_BGR2GRAY);
		fprintf(stderr,"HERE\n");
		//cv::Mat::copyTo(item->in_frame,gs_frame);
		//item->in_frame = gs_frame.clone();
	}
		fprintf(stderr,"HERE\n");

	return item;
}

int initThreadNode(TPN_t* td,int threadNum) {
    td->threadNum = threadNum;
    td->selfDestruct = false;
    td->isJoinable = false;
    td->taskType = -1; //no task
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

int getWorkFromTaskList(taskOb_t** task) {
    if(pthread_mutex_trylock(&G_listMutex) == 0) {
        //list is free for us to grab
        if(!G_taskList.empty()){
            *task = G_taskList.front();
            G_taskList.pop_front();
        }
        pthread_mutex_unlock(&G_listMutex);
    } else {return -1;}
    return 0;
}

int putInWorkProductList(taskOb_t* task) {
   	pthread_mutex_lock(&G_workProductMutex);
    //list is free for us to grab
    G_workProduct.push_back(task);
    pthread_mutex_unlock(&G_listMutex);
    return 0;
}

int getFromWorkProductList(taskOb_t** task) {
   	pthread_mutex_lock(&G_workProductMutex);
    if(!G_taskList.empty()){
        *task = G_taskList.front();
        G_taskList.pop_front();
    }
    pthread_mutex_unlock(&G_listMutex);
    return 0;
}

 int workerThreadLoop(TPN_t* tn) {
	//init variables
	cv::Mat white_mask;

	while(1) {
 	   //check if the thread has been asked to join
        if(pthread_mutex_trylock(&tn->mutex1)==0){
            //fprintf(stderr,"Hello from thread %i, I should be joinable? %d\n",
            	//tn->threadNum,tn->isJoinable);
            if(tn->isJoinable == true) {
                pthread_mutex_unlock(&tn->mutex1);
                return 0;
            } else {
                pthread_mutex_unlock(&tn->mutex1);
            }
        }
        //check for work to do from the queue
        taskOb_t* task = NULL;
        int err = getWorkFromTaskList(&task); //returns in task     if available
        if(err==0 && task != NULL) {
            switch(task->taskType) {
			case SIFT_TASK: {
	    		//std::vector<cv::KeyPoint> keypoints = SIFTonFrame(task->in_frame);
				break;
			}
			case CANNY_TASK: {
				//white_mask.create(task->in_frame.size(),task->in_frame.type());
				//cv::Mat detected_edges = cannyRoutine(task->in_frame,white_mask);
				//task->return_frame = detected_edges.clone();
				putInWorkProductList(task);
				break;
			}
			case HARRIS_TASK: {
				//std::vector<cv::Point> harrisResultsVector =
					//harrisCornerDetection(task->in_frame);
				break;
			}
			default: {
				fprintf(stderr,"Invalid task received at line %i\n",__LINE__);
			}
            }
			usleep(1);
        }
        //encourage scheduling of other threads
        usleep(1);
    }
    //should never reach
    return -1;
}


int processVideo(cv::VideoCapture cap) {
	long frameCount = 0;
	std::string windowName = "Output";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);

		fprintf(stderr,"H1ERE\n");
	cv::Mat detected_edges;
	while(true) {
		cv::Mat in_frame;
		cv::Mat out_frame;

		bool bSuccess = cap.read(in_frame);
		if(bSuccess == false) {
			fprintf(stderr,"EOF\n");
			break;
		}
		out_frame = in_frame.clone();

		//send frame out to worker queue for each applicable type
		pthread_mutex_lock(&G_listMutex);
		fprintf(stderr,"HE2RE\n");
		taskOb_t* task = newTaskOb(CANNY_TASK,frameCount,in_frame);
		fprintf(stderr,"HE3RE\n");
		G_taskList.push_back(task);
		fprintf(stderr,"HER4E\n");

		pthread_mutex_unlock(&G_listMutex);

		fprintf(stderr,"HERE\n");

		//spin on the return queue until we have the responses we expect
		task = NULL;
		while(!G_workProduct.empty()){
			getFromWorkProductList(&task);
			pthread_mutex_lock(&(task->mutex1));
			switch(task->taskType) {

			case SIFT_TASK: {
				break;
			}
			case CANNY_TASK: {
				//detected_edges = task->return_frame;
				break;
			}
			case HARRIS_TASK: {
				break;
			}

			default: {
				fprintf(stderr,"Invalid task received at line %i\n",__LINE__);
			}
			}
			pthread_mutex_unlock(&(task->mutex1));
			free(task);
            usleep(1); //encourage scheduling of other threads
		}

		fprintf(stderr,"HERE\n");

		//display the return items


		cv::Mat white_mask;
		cv::Mat gs_frame;

		white_mask.create(out_frame.size(),out_frame.type());
		cv::cvtColor(in_frame,gs_frame,cv::COLOR_BGR2GRAY);

		//SIFT
	    std::vector<cv::KeyPoint> keypoints = SIFTonFrame(in_frame);

		//corner detection
		std::vector<cv::Point> harrisResultsVector = harrisCornerDetection(gs_frame);

		//canny
		//detected_edges = cannyRoutine(gs_frame,white_mask);


		applyFrameMarking(white_mask,detected_edges,out_frame,keypoints,
						  harrisResultsVector);

		//output
		cv::imshow(windowName,out_frame);
		cv::waitKey(100); //delay TODO remove & use timer
	}


	return 0;
}




int main(int argc, char** argv )
{
	//begin ARG parsing
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

	//End ARG parsing
	//setup threadpool


	int hardwareThreads = std::thread::hardware_concurrency();
	fprintf(stderr,"threads %i\n\n",hardwareThreads);
	TPN_t** threadData = initThreadPoolData(hardwareThreads);

    for(int i = 0; i < hardwareThreads; i++) {
        initThreadNode(threadData[i],i);
        G_threads.emplace_back(workerThreadLoop,threadData[i]);
    }

	processVideo(cap);
	cv::waitKey(10000);
	internalShutdownPool(threadData,hardwareThreads);
	return 0;
}


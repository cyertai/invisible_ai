#Invisible AI Assement
Implementing a multi-threaded image processor using OpenCV

## Overview
Here I implement a multi-threaded image processor that takes a video file as its first argument, and computes interesting characteristics of the image, such as:

	Edges, detected using cv::Canny
	SIFT features, using cv::SIFT
	Corners, using cv::cornerHarris

The user steps through each frame using the keyboard; and the relevant features are computed in three seperate threads, belonging to the threadpool.

## Folder Structure and Execution
. contains CMakeLists.txt and the README.md
/src contains the project code
/data contains an example image, and is where you should store your target files
/bin contains compiled executables

To build, from the home directory run:

	cmake .
	cmake --build .

To run, from the bin directory, type

	./multiFeatureDetection <filepath>

## Code Design

### Algorithms
For each algorithm, I used the example settings and implementation from the openCV documentation, adjusted a little bit as time allotted.

### Structure
We start a main thread, which:

	Opens the video file
	Starts the threadpool
	Enters this loop:

		Loops over each frame in the video, sending tasks to the threadpool
		Checks the responding data structures for completed tasks
		Displays the results from processing

	Shuts down the threadpool once processing is completed
	Exits

Each worker thread exists in the following loop:

	Checks if it has been asked to join by the main thread
		If so, returns and implicit call to pthread_exit()

	Checks for work to do in in the shared task data structures
		Does work if it gets some
		Stores work in the appropriate response list


Data structures:

	There are several important data structures in this program:
		- threadData (TPN_t** - Thread Pool Node array)
		- G_taskListMutex (A global mutex for shared lists)
		- Global send and receive queues for work between threads

	The threadData structure allows the master thread to send a shutdown signal
	to the worker threads, cleanly joining with them.

	I used a pthread_mutex instead of a std::mutex because pthread_mutexes support

	pthread_mutex_trylock(), a non-blocking attempt to grab a mutex that allows
	threads to spin on several different data structures in a loop

	I intended to use a shared structure to communicate work tasks and cv::Mat
	data to and from the worker threads, but an early implementation of this ran
	into difficulties mallocing a structure containing a cv::Mat. In the interest
	of time I implemented global queues in the form of std::lists

### Desired Final Structure
While the code as is works and achieves the stated objectives, it falls short in several ways:

		- Inability to pre-compute results before the user advances the frame
		- Inability to utilize the supported number of machine threads (12 in my case)
		- Master thread is also the display thread, and is blocking

I desired, if I had the time, to make the following changes:

		- The master thread spins off frames to the task lists, keeping them with
		  a sufficient level of work to do at all times
		- Clean communication struct between master->worker->display threads
		  allowing for clean retrieval of pre-computed results
		- Display thread seperate from the master thread, so that computation
		  can progress without waiting for the user

I see several challenges in implementing this:

		- Managing the level of pre-computed results so that memory usage does
		  not become a challenge
		- Bookkeeping to match computed results with the correct display frame
		- Design of the thread communication object containing cv::Mat and living in the heap

## Challenges
I had several challenges with this project, and felt like I learned a lot

		- building opencv_contrib with the correct flags took some time and research
		- could not find a way to use cv::Mats allocated in the heap to share
		  data between threads
		- Aforementioned desired final structure that I wished to implement instead

## Time Spent
	 - Implementation ~ 6.5hr across several sessions
	 - This writeup ~ 1hr
	 - Flowchart ~ 1/2 hr


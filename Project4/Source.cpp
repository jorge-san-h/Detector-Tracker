// "Usage example: Project4.exe inoshishi_tiny.cfg inoshishi_tiny.weights inoshishi.names"
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>

#define OPENCV

#include "yolo_v2_class.hpp"	// imported functions from DLL

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>		
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
#include "opencv2/tracking.hpp"
#pragma comment(lib, "opencv_world342.lib")  

using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream >( ( std::ostringstream() << std::dec << x ) ).str()

// Initialize the parameters
vector<string> classes;
bool ok;
bool tracker_isactive;


static double diffclock(clock_t clock1, clock_t clock2)
{
	double diffticks = clock1 - clock2;
	double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
	return diffms;
}

void draw_boxes(cv::Mat& mat_img, std::vector<bbox_t> result_vec) {

	for (auto &i : result_vec) {

		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), cv::Scalar(0, 0, 255), 2);

		string label = cv::format("%.2f", i.prob);
		label = classes[i.obj_id] + ":" + label;

		//Display the label at the top of the bounding box
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int left = i.x;
		int top = i.y;
		top = max(top, labelSize.height);

		cv::rectangle(mat_img, cv::Point(left, top - 8 - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine - 8), cv::Scalar(255, 255, 255), cv::FILLED);
		cv::putText(mat_img, label, cv::Point(left, top - 6), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
	}
}

//void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
//	for (auto &i : result_vec) {
//		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
//		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
//			<< ", w = " << i.w << ", h = " << i.h
//			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
//	}
//}

int main(int argc, char** argv)
{
	Detector detector(argv[1], argv[2]);

	string classesFile = argv[3];
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	if (classes.empty())
	{
		cout << "Classes not found" << endl;
		return 1;
	}

	// FOR VIDEO

	//string str, outputFile;
	//cv::VideoCapture cap;
	//cv::VideoWriter video;
	//cv::Mat frame;
	//try {
	//	// Open the video file
	//	str = argv[4];
	//	ifstream ifile(str);
	//	if (!ifile) throw("error");
	//	cap.open(str);
	//	str.replace(str.end() - 4, str.end(), "_output.avi");
	//	outputFile = str;
	//}
	//catch (...) {
	//	cout << "Could not open the input video stream" << endl;
	//	return 0;
	//}

	// FOR WEBCAM 

	string outputFile = "webcam_output.avi";
	cv::VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
	{
		cout << "failed" << endl;
		cv::waitKey(3000);
		return -1;
	}
	cv::Mat frame, framegray;
	//activate to record video
	//cv::VideoWriter video;
	// Get the video writer initialized to save the output video
	//video.open(outputFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));


	cv::Ptr<cv::Tracker> tracker;
	tracker_isactive = false;
	cv::Rect2d roi;

	int chanceframes = 0;

	while (cv::waitKey(30) < 0)
	{
		//Starting FPS count
		double timer = (double)cv::getTickCount();
		
		cap >> frame;
		if (frame.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			cv::waitKey(3000);
			break;
		}

		std::vector<bbox_t> result_vec = detector.detect(frame, 0.2);

		if (!result_vec.empty()) {

			draw_boxes(frame, result_vec);

			cv::Rect2d roi((float)result_vec[0].x, (float)result_vec[0].y, (float)result_vec[0].w, (float)result_vec[0].h);
			tracker = cv::TrackerKCF::create();
			tracker->init(frame, roi);

			cv::putText(frame, "DETECTING", cv::Point(50, 130), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
			tracker_isactive = true;
		}
		else
		{
			if (tracker_isactive) {
				ok = tracker->update(frame, roi);
				if (ok) {
					// Tracking success: Draw the tracked object
					cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);
					cv::putText(frame, "TRACKING", cv::Point(50, 130), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 2);
				}
				else
				{
					// Tracking failure detected
					cv::putText(frame, "Tracking failure detected", cv::Point(50, 130), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 2);
					chanceframes++;
					if (chanceframes > 60)
					{
						tracker_isactive = false;
						chanceframes = 0;
					}
					
				}
			}
		}
		
		//Getting FPS count
		float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
		putText(frame, "FPS : " + SSTR(int(fps)), cv::Point(50, 70), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		
		cv::imshow("Output video", frame);

		//activate to record video
		//video.write(frame);   
	}

	cap.release();
	//activate to record video
	//video.release();

	return 0;
}
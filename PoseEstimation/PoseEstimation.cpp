#define _CRT_SECURE_NO_WARNINGS
#pragma warning(suppress : 4996)

#include<stdio.h>
#include<string>
#include<vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include<chrono>
#include<random>
#include<set>
#include<cmath>


#if 0

int main()
{

#if 0
	//
	const int POSE_PAIRS[14][2] =
	{
	 { 0,1 },{ 1,2 },{ 2,3 },
	 { 3,4 },{ 1,5 },{ 5,6 },
	 { 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
	 { 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
	};
	int nPoints = 15;


	// Specify the paths for the 2 files
	std::string protoFile = "pose_deploy_linevec_faster_4_stages.prototxt";
	std::string weightsFile = "pose_iter_160000.caffemodel";
#else
	const int POSE_PAIRS[17][2] =
	{
		{1,2}, {1,5}, {2,3},
		{3,4}, {5,6}, {6,7},
		{1,8}, {8,9}, {9,10},
		{1,11}, {11,12}, {12,13},
		{1,0}, {0,14},
		{14,16}, {0,15}, {15,17}
	};

	std::string protoFile = "pose_deploy_linevec.prototxt";
	std::string weightsFile = "pose_iter_440000.caffemodel";

	int nPoints = 18; 
#endif


	// Read the network into Memory
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);

	bool video_mode = false;
	cv::Mat color;
	cv::Mat frame, frameCopy;

#if 10
	video_mode = true;
	std::string videoFile = "sample_video.mp4";
	cv::VideoCapture cap(videoFile);
	if (!cap.isOpened())
	{
		std::cerr << "Unable to connect to camera" << std::endl;
		return 1;
	}
	int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	
	cv::VideoWriter video("Output-Skeleton.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frameWidth, frameHeight)); 
#else
	video_mode = false;

	color = cv::imread("multiple.jpeg");
	cv::cvtColor(color, frame, CV_BGRA2BGR);

	//
	cv::Mat frameCopy = frame.clone();
	int frameWidth = frame.cols;
	int frameHeight = frame.rows;
#endif


	// Specify the input image dimensions
	int inWidth = 368;
	int inHeight = 368;
	float thresh = 0.1;

	do
	{
		if (video_mode)
		{
			if (cv::waitKey(1) < 0)
			{
				cap >> frame;
				frameCopy = frame.clone();
			}
			else
			{
				break;
			}
		}

		// Prepare the frame to be fed to the network
		cv::Mat inpBlob = cv::dnn::blobFromImage(frame, 1.0 / 255, cv::Size(inWidth, inHeight), cv::Scalar(0, 0, 0), false, false);

		// Set the prepared object as the input blob of the network
		net.setInput(inpBlob);

		cv::Mat output = net.forward();


		int H = output.size[2];
		int W = output.size[3];

		// find the position of the body parts
		std::vector<cv::Point> points(nPoints);
		for (int n = 0; n < nPoints; n++)
		{
			// Probability map of corresponding body's part.
			cv::Mat probMap(H, W, CV_32F, output.ptr(0, n));

			cv::Point2f p(-1, -1);
			cv::Point maxLoc;
			double prob;
			cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
			if (prob > thresh)
			{
				p = maxLoc;
				p.x *= (float)frameWidth / W;
				p.y *= (float)frameHeight / H;

				cv::circle(frameCopy, cv::Point((int)p.x, (int)p.y), 8, cv::Scalar(0, 255, 255), -1);
				cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			}
			points[n] = p;
		}

		int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

		for (int n = 0; n < nPairs; n++)
		{
			// lookup 2 connected body/hand parts
			cv::Point2f partA = points[POSE_PAIRS[n][0]];
			cv::Point2f partB = points[POSE_PAIRS[n][1]];

			if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
				continue;

			cv::line(frame, partA, partB, cv::Scalar(0, 255, 255), 8);
			cv::circle(frame, partA, 8, cv::Scalar(0, 0, 255), -1);
			cv::circle(frame, partB, 8, cv::Scalar(0, 0, 255), -1);
		}

		if (video_mode)
		{
			imshow("Output-Skeleton", frame);
			video.write(frame);
		}
		else
		{
			cv::cvtColor(frame, color, CV_BGR2BGRA);
			cv::imwrite("result.png", frame);
		}
	} while (video_mode);

	if (video_mode)
	{
		cap.release();
		video.release();
	}

	return 0;
}
#else
////////////////////////////////

std::vector<cv::Scalar> colors;
const int ColorsMax = 10000;

struct KeyPoint {
	KeyPoint(cv::Point point, float probability) {
		this->id = -1;
		this->point = point;
		this->probability = probability;
	}

	int id;
	cv::Point point;
	float probability;
};

std::ostream& operator << (std::ostream& os, const KeyPoint& kp)
{
	os << "Id:" << kp.id << ", Point:" << kp.point << ", Prob:" << kp.probability << std::endl;
	return os;
}

////////////////////////////////
struct ValidPair {
	ValidPair(int aId, int bId, float score) {
		this->aId = aId;
		this->bId = bId;
		this->score = score;
	}

	int aId;
	int bId;
	float score;
};

std::ostream& operator << (std::ostream& os, const ValidPair& vp)
{
	os << "A:" << vp.aId << ", B:" << vp.bId << ", score:" << vp.score << std::endl;
	return os;
}

////////////////////////////////

template < class T > std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
	os << "[";
	bool first = true;
	for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
	{
		if (!first) os << ",";
		os << " " << *ii;
	}
	os << "]";
	return os;
}

template < class T > std::ostream& operator << (std::ostream& os, const std::set<T>& v)
{
	os << "[";
	bool first = true;
	for (typename std::set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
	{
		if (!first) os << ",";
		os << " " << *ii;
	}
	os << "]";
	return os;
}

////////////////////////////////

const int nPoints = 18;

const std::string keypointsMapping[] = {
	"Nose", "Neck",
	"R-Sho", "R-Elb", "R-Wr",
	"L-Sho", "L-Elb", "L-Wr",
	"R-Hip", "R-Knee", "R-Ank",
	"L-Hip", "L-Knee", "L-Ank",
	"R-Eye", "L-Eye", "R-Ear", "L-Ear"
};

const std::vector<std::pair<int, int>> mapIdx = {
	{31,32}, {39,40}, {33,34}, {35,36}, {41,42}, {43,44},
	{19,20}, {21,22}, {23,24}, {25,26}, {27,28}, {29,30},
	{47,48}, {49,50}, {53,54}, {51,52}, {55,56}, {37,38},
	{45,46}
};

const std::vector<std::pair<int, int>> posePairs = {
	{1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
	{1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
	{1,0}, {0,14}, {14,16}, {0,15}, {15,17}, {2,17},
	{5,16}
};

void getKeyPoints(cv::Mat& probMap, double threshold, std::vector<KeyPoint>& keyPoints) {
	cv::Mat smoothProbMap;
	cv::GaussianBlur(probMap, smoothProbMap, cv::Size(3, 3), 0, 0);

	cv::Mat maskedProbMap;
	cv::threshold(smoothProbMap, maskedProbMap, threshold, 255, cv::THRESH_BINARY);

	maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); ++i) {
		cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());

		cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

		double maxVal;
		cv::Point maxLoc;

		cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);

		keyPoints.push_back(KeyPoint(maxLoc, probMap.at<float>(maxLoc.y, maxLoc.x)));
	}
}

void populateColorPalette(std::vector<cv::Scalar>& colors, int nColors) {
	std::random_device rd;
	std::mt19937 gen(rd());
#if 0
	std::uniform_int_distribution<> dis1(64, 200);
	std::uniform_int_distribution<> dis2(100, 255);
	std::uniform_int_distribution<> dis3(100, 255);
#else
	std::uniform_int_distribution<> dis1(100, 255);
	std::uniform_int_distribution<> dis2(10, 255);
	std::uniform_int_distribution<> dis3(10, 255);
#endif

	for (int i = 0; i < nColors; ++i) {
		colors.push_back(cv::Scalar(dis1(gen), dis2(gen), dis3(gen)));
	}
}

void splitNetOutputBlobToParts(cv::Mat& netOutputBlob, const cv::Size& targetSize, std::vector<cv::Mat>& netOutputParts) {
	int nParts = netOutputBlob.size[1];
	int h = netOutputBlob.size[2];
	int w = netOutputBlob.size[3];

	for (int i = 0; i < nParts; ++i) {
		cv::Mat part(h, w, CV_32F, netOutputBlob.ptr(0, i));

		cv::Mat resizedPart;

		cv::resize(part, resizedPart, targetSize);

		netOutputParts.push_back(resizedPart);
	}
}

void populateInterpPoints(const cv::Point& a, const cv::Point& b, int numPoints, std::vector<cv::Point>& interpCoords) {
	float xStep = ((float)(b.x - a.x)) / (float)(numPoints - 1);
	float yStep = ((float)(b.y - a.y)) / (float)(numPoints - 1);

	interpCoords.push_back(a);

	for (int i = 1; i < numPoints - 1; ++i) {
		interpCoords.push_back(cv::Point(a.x + xStep * i, a.y + yStep * i));
	}

	interpCoords.push_back(b);
}


void getValidPairs(const std::vector<cv::Mat>& netOutputParts,
	const std::vector<std::vector<KeyPoint>>& detectedKeypoints,
	std::vector<std::vector<ValidPair>>& validPairs,
	std::set<int>& invalidPairs) {

	int nInterpSamples = 10;
	float pafScoreTh = 0.1;
	float confTh = 0.7;

	for (int k = 0; k < mapIdx.size(); ++k) {

		//A->B constitute a limb
		cv::Mat pafA = netOutputParts[mapIdx[k].first];
		cv::Mat pafB = netOutputParts[mapIdx[k].second];

		//Find the keypoints for the first and second limb
		const std::vector<KeyPoint>& candA = detectedKeypoints[posePairs[k].first];
		const std::vector<KeyPoint>& candB = detectedKeypoints[posePairs[k].second];

		int nA = candA.size();
		int nB = candB.size();

		/*
		  # If keypoints for the joint-pair is detected
		  # check every joint in candA with every joint in candB
		  # Calculate the distance vector between the two joints
		  # Find the PAF values at a set of interpolated points between the joints
		  # Use the above formula to compute a score to mark the connection valid
		 */

		if (nA != 0 && nB != 0) {
			std::vector<ValidPair> localValidPairs;

			for (int i = 0; i < nA; ++i) {
				int maxJ = -1;
				float maxScore = -1;
				bool found = false;

				for (int j = 0; j < nB; ++j) {
					std::pair<float, float> distance(candB[j].point.x - candA[i].point.x, candB[j].point.y - candA[i].point.y);

					float norm = std::sqrt(distance.first*distance.first + distance.second*distance.second);

					if (!norm) {
						continue;
					}

					distance.first /= norm;
					distance.second /= norm;

					//Find p(u)
					std::vector<cv::Point> interpCoords;
					populateInterpPoints(candA[i].point, candB[j].point, nInterpSamples, interpCoords);
					//Find L(p(u))
					std::vector<std::pair<float, float>> pafInterp;
					for (int l = 0; l < interpCoords.size(); ++l) {
						pafInterp.push_back(
							std::pair<float, float>(
								pafA.at<float>(interpCoords[l].y, interpCoords[l].x),
								pafB.at<float>(interpCoords[l].y, interpCoords[l].x)
								));
					}

					std::vector<float> pafScores;
					float sumOfPafScores = 0;
					int numOverTh = 0;
					for (int l = 0; l < pafInterp.size(); ++l) {
						float score = pafInterp[l].first*distance.first + pafInterp[l].second*distance.second;
						sumOfPafScores += score;
						if (score > pafScoreTh) {
							++numOverTh;
						}

						pafScores.push_back(score);
					}

					float avgPafScore = sumOfPafScores / ((float)pafInterp.size());

					if (((float)numOverTh) / ((float)nInterpSamples) > confTh) {
						if (avgPafScore > maxScore) {
							maxJ = j;
							maxScore = avgPafScore;
							found = true;
						}
					}

				}/* j */

				if (found) {
					localValidPairs.push_back(ValidPair(candA[i].id, candB[maxJ].id, maxScore));
				}

			}/* i */

			validPairs.push_back(localValidPairs);

		}
		else {
			invalidPairs.insert(k);
			validPairs.push_back(std::vector<ValidPair>());
		}
	}/* k */
}

void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>>& validPairs,
	const std::set<int>& invalidPairs,
	std::vector<std::vector<int>>& personwiseKeypoints) {
	for (int k = 0; k < mapIdx.size(); ++k) {
		if (invalidPairs.find(k) != invalidPairs.end()) {
			continue;
		}

		const std::vector<ValidPair>& localValidPairs(validPairs[k]);

		int indexA(posePairs[k].first);
		int indexB(posePairs[k].second);

		for (int i = 0; i < localValidPairs.size(); ++i) {
			bool found = false;
			int personIdx = -1;

			for (int j = 0; !found && j < personwiseKeypoints.size(); ++j) {
				if (indexA < personwiseKeypoints[j].size() &&
					personwiseKeypoints[j][indexA] == localValidPairs[i].aId) {
					personIdx = j;
					found = true;
				}
			}/* j */

			if (found) {
				personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
			}
			else if (k < 17) {
				std::vector<int> lpkp(std::vector<int>(18, -1));

				lpkp.at(indexA) = localValidPairs[i].aId;
				lpkp.at(indexB) = localValidPairs[i].bId;

				personwiseKeypoints.push_back(lpkp);
			}

		}/* i */
	}/* k */
}


int main(int argc, char** argv) 
{
	bool video_mode = false;
	cv::VideoCapture cap;
	std::string inputFile;
	cv::Mat input;
	cv::Mat inputCopy;

	cv::VideoWriter video;
	//if (argc <= 1) 
	//{
	//	return 0;
	//}

	if (video_mode)
	{
		std::string videoFile = "sample_video.mp4";
		cap = cv::VideoCapture(videoFile);
		if (!cap.isOpened())
		{
			std::cerr << "Unable to connect to camera" << std::endl;
			return 1;
		}
		int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

		video = cv::VideoWriter("result_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frameWidth, frameHeight));
	}
	else
	{
		inputFile = "multiple.jpeg";
		input = cv::imread(inputFile, cv::IMREAD_COLOR);
	}
	cv::dnn::Net inputNet = cv::dnn::readNetFromCaffe("pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel");

	// Specify the input image dimensions
	int inWidth = 368;
	int inHeight = 368;
	float thresh = 0.1;



	std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();


	do
	{
		if (video_mode)
		{
			if (cv::waitKey(1) < 0)
			{
				cap >> input;
			}
			else
			{
				break;
			}
		}

		cv::Mat inputBlob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size((int)((inWidth * input.cols) / input.rows), inHeight), cv::Scalar(0, 0, 0), false, false);

		inputNet.setInput(inputBlob);

		cv::Mat netOutputBlob = inputNet.forward();

		std::vector<cv::Mat> netOutputParts;
		splitNetOutputBlobToParts(netOutputBlob, cv::Size(input.cols, input.rows), netOutputParts);

		std::chrono::time_point<std::chrono::system_clock> finishTP = std::chrono::system_clock::now();

		std::cout << "Time Taken in forward pass = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP - startTP).count() << " ms" << std::endl;

		int keyPointId = 0;
		std::vector<std::vector<KeyPoint>> detectedKeypoints;
		std::vector<KeyPoint> keyPointsList;

		for (int i = 0; i < nPoints; ++i) {
			std::vector<KeyPoint> keyPoints;

			getKeyPoints(netOutputParts[i], 0.1, keyPoints);

			std::cout << "Keypoints - " << keypointsMapping[i] << " : " << keyPoints << std::endl;

			for (int i = 0; i < keyPoints.size(); ++i, ++keyPointId) {
				keyPoints[i].id = keyPointId;
			}

			detectedKeypoints.push_back(keyPoints);
			keyPointsList.insert(keyPointsList.end(), keyPoints.begin(), keyPoints.end());
		}

#if 0
		//std::vector<cv::Scalar> colors;
		//populateColorPalette(colors, nPoints);
#else		
		if (colors.size() == 0)
		{
			populateColorPalette(colors, ColorsMax);
		}
#endif
		cv::Mat outputFrame = input.clone();

		for (int i = 0; i < nPoints; ++i) {
			for (int j = 0; j < detectedKeypoints[i].size(); ++j) {
				cv::circle(outputFrame, detectedKeypoints[i][j].point, 5, colors[i], -1, cv::LINE_AA);
			}
		}

		std::vector<std::vector<ValidPair>> validPairs;
		std::set<int> invalidPairs;
		getValidPairs(netOutputParts, detectedKeypoints, validPairs, invalidPairs);

		std::vector<std::vector<int>> personwiseKeypoints;
		getPersonwiseKeypoints(validPairs, invalidPairs, personwiseKeypoints);

		for (int i = 0; i < nPoints - 1; ++i) {
			for (int n = 0; n < personwiseKeypoints.size(); ++n) {
				const std::pair<int, int>& posePair = posePairs[i];
				int indexA = personwiseKeypoints[n][posePair.first];
				int indexB = personwiseKeypoints[n][posePair.second];

				if (indexA == -1 || indexB == -1) {
					continue;
				}

				const KeyPoint& kpA = keyPointsList[indexA];
				const KeyPoint& kpB = keyPointsList[indexB];

				cv::line(outputFrame, kpA.point, kpB.point, colors[i], 3, cv::LINE_AA);

			}
		}

		if (video_mode)
		{
			imshow("result_video.png", outputFrame);
			video.write(outputFrame);
			cv::waitKey(1);
		}
		else
		{
			cv::Mat frame;
			//cv::cvtColor(frame, outputFrame, CV_BGR2BGRA);
			cv::imwrite("result.png", outputFrame);
			cv::waitKey(1);
		}

	} while (video_mode);

	return 0;
}

#endif
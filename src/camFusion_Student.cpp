
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(100); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> kptMatchesRoi; 
    for (auto match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            kptMatchesRoi.push_back(match); 
        }
    }
    if (kptMatchesRoi.empty()) 
        return; 

    // filter matches 
    double accumulatedDist = 0.0; 
    for  (auto it = kptMatchesRoi.begin(); it != kptMatchesRoi.end(); ++it)  
         accumulatedDist += it->distance; 
    double meanDist = accumulatedDist / kptMatchesRoi.size();  
    double threshold = meanDist / 0.7;        
    for  (auto it = kptMatchesRoi.begin(); it != kptMatchesRoi.end(); ++it)
    {
       if (it->distance < threshold)
           boundingBox.kptMatches.push_back(*it);
    }
    cout << "Leave " << boundingBox.kptMatches.size()  << " matches" << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    cout << "distRatios size=" << distRatios.size() << endl;
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);

    cout << "medDistRatio=" << medDistRatio << endl; 
    cout << "TTC measured by camera = " << TTC << endl;
}


double getMedianXDirection(std::vector<LidarPoint> &lidarPoints) {
    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint &pt1, LidarPoint &pt2) {
        return pt1.x < pt2.x;
    });
    if (lidarPoints.empty()) 
        return NAN; 
    int medianIdx = floor(lidarPoints.size() / 2.0); 
    double medianX = lidarPoints.size() % 2 == 0 ? (lidarPoints[medianIdx - 1].x + lidarPoints[medianIdx].x) / 2.0 : lidarPoints[medianIdx].x; 
    return lidarPoints[lidarPoints.size() / 2].x; 
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    cout << "lidarPointsPrev size=" << lidarPointsPrev.size() << endl;
    cout << "lidarPointsCurr size=" << lidarPointsCurr.size() << endl;

    double medianXPrev = getMedianXDirection(lidarPointsPrev);
    double medianXCurr = getMedianXDirection(lidarPointsCurr);
    double dT = 1.0 / frameRate;

    cout << "medianXPrev=" << medianXPrev << endl;
    cout << "medianXCurr=" << medianXCurr << endl;

    TTC = medianXCurr * dT / (medianXPrev - medianXCurr);
    cout << "TTC measured by lidar = " << TTC << endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // method 1:
    // std::map<int, std::multiset<int>> multiBBmatches;
    // for (const auto& match : matches) {
    //     cv::KeyPoint prevKpt = prevFrame.keypoints[match.queryIdx];
    //     cv::KeyPoint currKpt = currFrame.keypoints[match.trainIdx]; 
    //     std::vector<int> prevIds, currIds; 
    //     for (const auto& box : prevFrame.boundingBoxes) {
    //         if (box.roi.contains(prevKpt.pt))
    //             prevIds.push_back(box.boxID); 
    //     }
    //     for (const auto& box : currFrame.boundingBoxes) {
    //         if (box.roi.contains(prevKpt.pt))
    //             currIds.push_back(box.boxID); 
    //     }
        
    //     for (auto prevId : prevIds) {
    //         for (auto currId : currIds) {
    //             // finding all matched bounding boxes if both bb for prev and curr frame contain matched keypoint 
    //             multiBBmatches[prevId].insert(currId);
    //         }
    //     }
    // }

    // for (auto bbMatches : multiBBmatches) {
    //     int bestCurrId = 0, maxCount = 0;
    //     // for each matched bb in previous frame, choose the best-bb in current frame that contains the most number of matched keypoints 
    //     for (auto it = bbMatches.second.begin(); it != bbMatches.second.end(); ++it) {
    //         int count = bbMatches.second.count(*it);
    //         if (count > maxCount) {
    //             bestCurrId = *it; 
    //             maxCount = count;
    //         }

    //     }
    //     if (maxCount) {
    //         bbBestMatches.insert({bbMatches.first, bestCurrId}); 
    //         std::cout << "match times: " << maxCount << " prevId: " << bbMatches.first << " currId: " << bestCurrId << std::endl;
    //     }
    // }


    // method 2:
    // loop for bounding box in previous frame 
    for (const auto& prevBb : prevFrame.boundingBoxes) {
        // find related bounding boxes in current frame that both bb in previous frame and current frame contain same keypoint
        std::multiset<int> currBbIds; 
        // loop for matched keypoints 
        for (const auto& match : matches) {
            cv::KeyPoint prevKpt = prevFrame.keypoints[match.queryIdx];
            cv::KeyPoint currKpt = currFrame.keypoints[match.trainIdx]; 
            if (!prevBb.roi.contains(prevKpt.pt)) {
                continue; // skip this match if the prevBb doesn't contain this keypoint
            }
            // loop for bounding box in current frame 
            for (const auto& currBb : currFrame.boundingBoxes) {
                if (!currBb.roi.contains(currKpt.pt)) {
                    continue; // skip this bb if currBb doesn't contain this keypoint
                }
                currBbIds.insert(currBb.boxID); 
            }

        }

        // find the best match bounding box in current frame that contains the highest number of matched keypoint
        int bestCurrId = 0, maxCount = 0;
        for (auto it = currBbIds.begin(); it != currBbIds.end(); ++it) {
            int count = currBbIds.count(*it);
            if (count > maxCount) {
                bestCurrId = *it; 
                maxCount = count;
            }
        }
        if (maxCount) {
            bbBestMatches.insert({prevBb.boxID, bestCurrId}); 
            std::cout << "match times: " << maxCount << " prevId: " << prevBb.boxID << " currId: " << bestCurrId << std::endl;
        }
    }

}

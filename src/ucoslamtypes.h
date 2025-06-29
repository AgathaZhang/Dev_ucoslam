/**
* This file is part of  UCOSLAM
*
* Copyright (C) 2018 Rafael Munoz Salinas <rmsalinas at uco dot es> (University of Cordoba)
*
* UCOSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* UCOSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with UCOSLAM. If not, see <http://wwmap->gnu.org/licenses/>.
*/
#ifndef __UCOSLAMTYPES_H
#define __UCOSLAMTYPES_H
#include <vector>
#include <cstdint>
#include <string>
#include <iostream>
#include <limits>
#include "ucoslam_exports.h"
#include <opencv2/core/core.hpp>
namespace ucoslam {

//states in which the system can be
enum STATE{STATE_TRACKING,STATE_LOST};  // STATE_TRACKING: the system is tracking the environment and can add new keyframes and points. STATE_LOST: the system has lost the tracking and cannot add new keyframes or points until it is reinitialized.
//working modes of the system
enum MODES{MODE_SLAM,MODE_LOCALIZATION};// MODE_SLAM: the system is in SLAM mode, adding keyframes and points to the map. MODE_LOCALIZATION: the system is in localization mode, trying to estimate the pose of the camera in a known map without adding new keyframes or points.
// MODES_SLAM: 系统处于 SLAM 模式，添加关键帧和点到地图中。 MODE_LOCALIZATION: 系统处于定位模式，尝试在已知地图中估计相机的位姿，而不添加新的关键帧或点。

/**
 * @brief The DescriptorTypes class defines types of descriptors that can be used for keypoints
 */

class UCOSLAM_API DescriptorTypes  {
public:
    //types of descriptors supported
    enum Type: std::int8_t {DESC_NONE=0, DESC_ORB=1,DESC_AKAZE=2,DESC_BRISK=3,DESC_FREAK=4,DESC_SURF=5,DESC_SIFT=6};

//converts from descriptor type to string
   static std::string toString(DescriptorTypes::Type type){
        switch (type)
        {
        case DESC_ORB: return "orb";
        case DESC_AKAZE: return "akaze";
        case DESC_BRISK: return "brisk";
        case DESC_FREAK: return "freak";
        case DESC_SURF: return "surf";
        default:
            throw std::runtime_error("DescriptorType::toString invalid descriptor ");
        }
    }
   //converts from string to descriptor type

   static DescriptorTypes::Type fromString(const std::string & strin){
       std::string str(strin);
       for(auto &c:str)c=tolower(c);
        if (str=="orb")return DESC_ORB;
        if (str=="akaze")return DESC_AKAZE;
        if (str=="brisk")return DESC_BRISK;
        if (str=="freak")return DESC_FREAK;
        if (str=="surf")return DESC_SURF;
        if (str=="sift")return DESC_SIFT;
        throw std::runtime_error("DescriptorType::fromString invalid descriptor string :"+str);
    }


};


/**
 * @brief The Params struct defines the main parameters for controlling the behaviour of the
 * UcoSlam library
 */

struct UCOSLAM_API Params{

    Params();

    //configure the appropriate parameters for the employed mode.
    //By default, non-sequential and orb is employed.
    //
    void setParams( bool sequential, DescriptorTypes::Type desc=DescriptorTypes::DESC_ORB);
    //////////////////////////////////////////
    ///Params interesting for the users
    //////////////////////////////////////////
    bool runSequential=true;//avoid parallel processing
    bool detectMarkers;//(dis/en)ables marker detection
    bool detectKeyPoints;//(dis/en)ables kp detection
    DescriptorTypes::Type kpDescriptorType;//keypoint descriptor employed
    bool KPNonMaximaSuppresion;//activating it will create maps with less points (faster).
    bool enableLoopClosure=true;//if disabled, loop closure detection and correction will not be enabled.
    float KFMinConfidence;//value that regulates when a keyframe is added. Range (0,inf). low values will include less keyframes. high value will include more keyframes
    int maxFeatures;//number of features to be detected in the image
    int nOctaveLevels;//number of octaves for keypoint detection
    float scaleFactor;//scale factor employed to create the octaves
    float KFCulling;// Value indicating how many redundant keypoints must be in a keyframe to remove it. Range [0,1]. If low, few keyframes will survive. If 1, no keyframe will be removed once added.
    float aruco_markerSize;//Size of markers in meters
    int maxNewPoints;//maximum number of new points created when a new keyframe is added
    bool reLocalizationWithKeyPoints=true;
    bool reLocalizationWithMarkers=true;
    bool inPlaneMarkers=false;//special case in which all markers are in the same plane.
    bool saveImageInMap=false;//saves in the keyframes of the map the compressed jpeg images. This will consumes tons of memory!



    bool forceInitializationFromMarkers;//If true, the system will not initialize until a good initialization from markers is obtained
    int nthreads_feature_detector;//number of threads employed in keypoint detection
    float markersOptWeight=0.5;// maximum importance of markers in the final error. Value in range [0,1]. The rest if assigned to points
    int minMarkersForMaxWeight=5;//how many markers are required to achieve assign the maximum weight to markers in the optimization process
    float kptImageScaleFactor=1;//[0,1] indicates the desired scale factor employed for keypoint detection.  If 1 the original input image is used. Otherwise, the
    //input image is resized with the specified scale factor. PLease notice that markers will be detected in the  original input images anyway



    bool autoAdjustKpSensitivity=false;//enables/disables automatic keypoint detector sensitivity to adapt environement with low texture
    //aruco::MarkerDetector::Params aruco_DetectorParams;//the internal parameters of Aruco Library for marker detection
    //parameters of the aruco marker detector
    std::string aruco_Dictionary="ARUCO_MIP_36h12";
    std::string aruco_DetectionMode="DM_NORMAL";
    std::string aruco_CornerRefimentMethod="CORNER_SUBPIX";
    float aruco_minMarkerSize=0;






    //////////////////////////////////////////
    ///Params not mean to be modified by users
    //////////////////////////////////////////
    std::string extraParams;
    float maxDescDistance=std::numeric_limits<float>::max();//maximum distance between descriptors to consider a possible match. It is internally rewritten according to the descriptor employed


    float baseline_medianDepth_ratio_min=0.01;
    std::string global_optimizer= "g2o";//which global optimizer to use
    int minNumProjPoints=3;//minimum number of keyframes in which a point must be seen to keep it
    int projDistThr;//when searching for points by projection, maximum 2d distance for search radius
    int maxVisibleFramesPerMarker;

    int aruco_minNumFramesRequired=3;            //minimum number of frames
    float aruco_minerrratio_valid=3;//minimum error ratio between two solutions to consider a initial pose valid
    bool aruco_allowOneFrameInitialization=false;
    float targetFocus=-1;//value to enable normalization across different cameras and resolutions


    //possible unused
    float thRefRatio=0.9;//ratoi of matches found in current frame compared to ref keyframe to consider a new keyframe to be inserted
     float minBaseLine=0.07;//minimum preffered distance  between keyframes. this value is ignored if no markers are present
#pragma warning "to remove "
    bool removeKeyPointsIntoMarkers=true;

    void toStream(std::ostream &str)const;
    void fromStream(std::istream &str);
    uint64_t getSignature()const;


    void saveToYMLFile(const std::string &path);
    void readFromYMLFile(const std::string &path);

private:
    template<typename Type>
    void attemtpRead(const std::string &name,Type &var,cv::FileStorage&fs ){
        if ( fs[name].type()!=cv::FileNode::NONE)
            fs[name]>>var;
    }

};




}

#endif

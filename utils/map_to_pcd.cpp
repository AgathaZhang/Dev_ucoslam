#include <iostream>
#include <fstream>
#include <string>
#include "map.h"

using namespace std;
using namespace ucoslam;

void saveMapPointsAndMarkersToPCD(const Map &map, const string &outputFilePath) {
    // 打开输出文件
    ofstream pcdFile(outputFilePath, ios::binary);
    if (!pcdFile.is_open()) {
        throw runtime_error("Cannot open output PCD file: " + outputFilePath);
    }

    // 获取地图点
    const auto &mapPoints = map.map_points;
    const auto &mapMarkers = map.map_markers;

    // 准备点云数据
    vector<cv::Vec4f> points;

    // // 添加地图点
    // for (const auto &mapPoint : mapPoints) {
    //     if (!mapPoint.isBad()) {
    //         cv::Point3f pos = mapPoint.getCoordinates();
    //         // 设置地图点颜色为红色 (RGB = 255, 0, 0)
    //         float redColor = 0xFF0000; // 红色的 RGB 值编码为浮点数
    //         points.emplace_back(pos.x, pos.y, pos.z, redColor);
    //     }
    // }

    // 添加 Marker 信息
    for (const auto &markerPair : mapMarkers) {
        const Marker &marker = markerPair.second;
        if (marker.pose_g2m.isValid()) {
            // 获取 Marker 的 3D 点
            auto markerPoints = marker.get3DPoints(true);
            for (const auto &point : markerPoints) {
                // 设置 Marker 点颜色为蓝色 (RGB = 0, 0, 255)
                float blueColor = 0xFF; // 蓝色的 RGB 值编码为浮点数
                points.emplace_back(point.x, point.y, point.z, blueColor);
            }
        }
    }

    // 写入 PCD 文件头
    pcdFile << "# .PCD v.7 - Point Cloud Data file format\n";
    pcdFile << "VERSION .7\n";
    pcdFile << "FIELDS x y z rgb\n";
    pcdFile << "SIZE 4 4 4 4\n";
    pcdFile << "TYPE F F F F\n";
    pcdFile << "COUNT 1 1 1 1\n";
    pcdFile << "WIDTH " << points.size() << "\n";
    pcdFile << "HEIGHT 1\n";
    pcdFile << "POINTS " << points.size() << "\n";
    pcdFile << "DATA binary\n";

    // 写入点云数据
    pcdFile.write(reinterpret_cast<const char *>(points.data()), points.size() * sizeof(cv::Vec4f));

    cout << "PCD file saved to: " << outputFilePath << endl;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_map_file> <output_pcd_file>" << endl;
        return -1;
    }

    string inputMapFile = argv[1];
    string outputPCDFile = argv[2];

    try {
        // 读取地图
        Map map;
        map.readFromFile(inputMapFile);

        // 保存点云到 PCD 文件
        saveMapPointsAndMarkersToPCD(map, outputPCDFile);
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
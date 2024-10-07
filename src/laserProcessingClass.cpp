
#include "laserProcessingClass.h"
#include "orbextractor.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <unordered_set> // 添加這行頭文件  `

//多threads
#include <thread>
#include <vector>
#include <mutex>

#include "opencv2/img_hash.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>


void LaserProcessingClass::init(lidar::Lidar lidar_param_in){
    lidar_param = lidar_param_in;
}


void LaserProcessingClass::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out){
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);
 
}

//surface===============================

void processImageRegions_surface(const cv::Mat& depthImage, cv::Mat gray, int startY, int endY, int half_window_size,
 double depth_threshold, double gradient_threshold, int intensity_threshold, int window_size, std::vector<cv::Point>& planePixels) {
    for (int y = startY + half_window_size; y < endY - half_window_size; y++) {
        for (int x = half_window_size; x < depthImage.cols - half_window_size; x++) {

                                // Check intensity
                bool is_same_intensity = true;
                uchar reference_intensity = gray.at<uchar>(y, x);
                for (int wx = -half_window_size*1; wx <= half_window_size*1; ++wx) {
                    for (int wy = -half_window_size * 1; wy <= half_window_size * 1; ++wy) {
                        if (std::abs(gray.at<uchar>(y + wy, x + wx) - reference_intensity) > intensity_threshold) {
                            is_same_intensity = false;
                            break;
                        }
                    }
                    if (!is_same_intensity) break;
                }

                if (is_same_intensity) {
                    if (depthImage.at<uchar>(y, x) != 0) {
                        planePixels.push_back(cv::Point(x, y));
                        //  // 輸出 planePixel 的資訊
                        //     std::cout << "Added plane pixel: (x = " << x << ", y = " << y << ")" << std::endl;
                        continue;
                    }
                }
            
        }
    }
}


void processImage_surface(const cv::Mat& depthImage,cv::Mat gray, int half_window_size,
 double depth_threshold, double gradient_threshold , int intensity_threshold, int window_size,
  std::vector<cv::Point>& planePixels) {
    int numThreads = 16; // Number of threads to use
    std::vector<std::thread> threads;
    std::vector<std::vector<cv::Point>> planePixelsList(numThreads);

    // Split the image into regions and create threads
    int totalHeight = depthImage.rows - depthImage.rows / 3.2; // Total height to process
    int heightPerThread = totalHeight / numThreads;
    int remainingHeight = totalHeight % numThreads;
    int startY = depthImage.rows / 3.2; // Start from one quarter of the image
    for (int i = 0; i < numThreads; ++i) {
        int endY = startY + heightPerThread;
        if (i == numThreads - 1) endY += remainingHeight;
        threads.emplace_back(processImageRegions_surface, std::cref(depthImage),gray, startY, endY, half_window_size, depth_threshold,gradient_threshold, intensity_threshold, window_size, std::ref(planePixelsList[i]));
        startY = endY;
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Combine results from each thread if needed
    for (const auto& pixels : planePixelsList) {
        planePixels.insert(planePixels.end(), pixels.begin(), pixels.end());
    }
}

//==============kd-tree & 內射法=======================

void performNearestNeighborInterpolation(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& interpolated_cloud,
    int num_interpolations,
    int& interpolated_points
) {

    // 開始計算 KD-Tree 構建時間
    auto kd_tree_start = std::chrono::high_resolution_clock::now();
    // 建立 KD-Tree
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(pc_out_surf);
    // 結束計算 KD-Tree 構建時間
    auto kd_tree_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kd_tree_time = kd_tree_end - kd_tree_start;

    ROS_INFO("KD-Tree building time: %f ms", kd_tree_time.count());

    interpolated_points = 0;

    // 遍歷點雲並進行內插
    for (size_t i = 0; i < pc_out_surf->points.size() - 1; ++i) {
        const pcl::PointXYZI& current_point = pc_out_surf->points[i];
        const pcl::PointXYZI& next_point = pc_out_surf->points[i + 1];

        // 計算兩點之間的距離
        double distance = std::sqrt(
            std::pow(next_point.x - current_point.x, 2) +
            std::pow(next_point.y - current_point.y, 2) +
            std::pow(next_point.z - current_point.z, 2)
        );

        if (distance < 1.5f) { // 使用合適的半徑
            // 插入內插點
            for (int m = 1; m <= num_interpolations; ++m) {
                double alpha = static_cast<double>(m) / (num_interpolations + 1);
                pcl::PointXYZI interp_point;
                interp_point.x = (1 - alpha) * current_point.x + alpha * next_point.x;
                interp_point.y = (1 - alpha) * current_point.y + alpha * next_point.y;
                interp_point.z = (1 - alpha) * current_point.z + alpha * next_point.z;
                interp_point.intensity = (1 - alpha) * current_point.intensity + alpha * next_point.intensity;

                // 查找內插點的最近鄰（k = 1）
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                if (kdtree.nearestKSearch(interp_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                    pcl::PointXYZI nearest_point = pc_out_surf->points[pointIdxNKNSearch[0]];
                    interpolated_cloud->points.push_back(nearest_point);
                    interpolated_points++;
                }
            }
        }
    }

    // 將內插點添加到原點雲
    *pc_out_surf += *interpolated_cloud;
    std::cout << "Total interpolated points: " << interpolated_points << std::endl;
}


//====================================================


void LaserProcessingClass::pointcloudtodepth(pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                                             sensor_msgs::ImageConstPtr& image_msg, 
                                             Eigen::Matrix<double, 3, 4>& matrix_3Dto2D,
                                             Eigen::Matrix3d& result,
                                             Eigen::Matrix3d& RR,
                                             Eigen::Vector3d& tt,
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first,
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf
                                             ){
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);

    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat depthImage = cv::Mat::zeros(gray.size(), CV_8UC1);
    cv::Mat depth_store = cv::Mat::zeros(gray.size(), CV_64FC1);

    double scale = (double)87/256;
    // double nani = 0;
    for (int i = 0; i < (int) surf_first->points.size(); i++) {
        if( surf_first->points[i].x > 0){
            double t = surf_first->points[i].x / scale;

            if (t > 255) {
                t = 255;
            }

            Eigen::Vector4d curr_point(surf_first->points[i].x, surf_first->points[i].y, surf_first->points[i].z, 1);
            Eigen::Vector3d curr_point_image = matrix_3Dto2D * curr_point;

            curr_point_image.x() = curr_point_image.x() / curr_point_image.z();
            curr_point_image.y() = curr_point_image.y() / curr_point_image.z();

            int x = static_cast<int>(curr_point_image.x());
            int y = static_cast<int>(curr_point_image.y());
            if (x >= 0 && x < depthImage.cols && y >= 0 && y < depthImage.rows) {
                depthImage.at<uchar>(y, x) = static_cast<uchar>(t);
                depth_store.at<double>(y, x) = curr_point_image.z();
                // nani++;
            }
        }
    }
    std::vector<cv::Point> planePixels;

    cv::Mat blurred , laplacian;
    
    // 使用高斯模糊
    cv::GaussianBlur(gray, gray , cv::Size(3, 3), 1.0);

    // 計算拉普拉斯變換
    // cv::Laplacian(blurred, laplacian, CV_16S, 5);

    // // 轉換為絕對值並轉換為 8 位無符號整數
    // cv::convertScaleAbs(laplacian, gray);

    // blur(gray, gray, cv::Size(3, 3));
    // cv::imshow("after", gray);
    // cv::waitKey(0);

    
    //***********************先看深度值 再看強度值***********************
    int window_size = 5;
    int intensity_threshold = 18; // 假設閾值為10
    int half_window_size = window_size / 2;
    double depth_threshold = 1;
    double gradient_threshold = 1.5;

processImage_surface(depthImage,gray, half_window_size, depth_threshold, gradient_threshold, intensity_threshold, window_size, planePixels);
 

    for (const cv::Point& point : planePixels) {
        int x = point.x;
        int y = point.y;
        double depth_value = depth_store.at<double>(y, x); // 從深度圖像中獲取深度值，注意型態為double
        
        if(depth_value != 0){
            Eigen::Vector3d points_3d(x*depth_value, y*depth_value, depth_value);
            Eigen::Vector3d recover;
            recover = RR.inverse() * (result*points_3d-tt);

            pcl::PointXYZI pcl_point;
            pcl_point.x = recover.x();
            pcl_point.y = recover.y();
            pcl_point.z = recover.z();

            // 輸出 pcl_point 的座標資訊
            // std::cout << "Added point: (x = " << pcl_point.x 
            //         << ", y = " << pcl_point.y 
            //         << ", z = " << pcl_point.z << ")" << std::endl;

            pc_out_surf->push_back(pcl_point);
            

            // Eigen::Vector4d curr_point(recover.x(), recover.y(), recover.z(), 1);
            // Eigen::Vector3d curr_point_image = matrix_3Dto2D * curr_point;

            // curr_point_image.x() = curr_point_image.x() / curr_point_image.z();
            // curr_point_image.y() = curr_point_image.y() / curr_point_image.z();

            // int xx = static_cast<int>(curr_point_image.x());
            // int yy = static_cast<int>(curr_point_image.y());

            // if (xx >= 0 && xx < depthImage.cols && yy >= 0 && yy < depthImage.rows) {
            //     cv::circle(cv_ptr->image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
            // }
        }
    }

    int num_interpolations = 1; // 每對點之間插入的內插點數量

    // 內插點雲
    pcl::PointCloud<pcl::PointXYZI>::Ptr interpolated_cloud(new pcl::PointCloud<pcl::PointXYZI>());

    // 開始時間
    ros::Time start_time = ros::Time::now();
    int interpolated_points ;
    // 執行內插並獲取內插點數量
    performNearestNeighborInterpolation(pc_out_surf, interpolated_cloud, num_interpolations,interpolated_points);
    
    // 結束時間
    ros::Time end_time = ros::Time::now();
    double processing_time = (end_time - start_time).toSec() * 1000.0; // 轉換為毫秒

    // 累積總處理時間和內插點數量
    static double total_interpolation_time = 0.0;
    static int total_frames = 0;

    total_interpolation_time += processing_time;
    total_frames++;

    // 計算並記錄平均時間
    double average_time = total_interpolation_time / total_frames;
    ROS_INFO("Average Nearest Neighbor Interpolation processing time: %f ms", average_time);

    // 將內插點添加到原點雲
    *pc_out_surf += *interpolated_cloud;
    double map_resolution = 0.2;
    downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2, map_resolution * 2);
    // downSamplingToMap(pc_out_surf, pc_out_surf);
    std::cout << "after plane number = " << pc_out_surf->points.size() << std::endl;

    // for (int i = 0; i < (int)pc_out_surf->points.size(); i++) {
    //     if (pc_out_surf->points[i].x >= 0) {
    //         Eigen::Vector4d curr_point(pc_out_surf->points[i].x, pc_out_surf->points[i].y, pc_out_surf->points[i].z, 1);
    //         Eigen::Vector3d curr_point_image = matrix_3Dto2D * curr_point;

    //         curr_point_image.x() = curr_point_image.x() / curr_point_image.z();
    //         curr_point_image.y() = curr_point_image.y() / curr_point_image.z();

    //         // 檢查投影點是否在邊緣上
    //         int x = static_cast<int>(curr_point_image.x());
    //         int y = static_cast<int>(curr_point_image.y());

    //         // 確保點在圖像範圍內
    //         if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
    //             cv::circle(cv_ptr->image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
    //         }
    //     }
    // }

    // // 顯示檢測到的中心像素值
    // cv::imshow("after plane", cv_ptr->image);
    // cv::waitKey(0);
}

void LaserProcessingClass::featureExtraction(pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, 
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge, 
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_first,
                                             sensor_msgs::ImageConstPtr& image_msg, 
                                             Eigen::Matrix<double, 3, 4>& matrix_3Dto2D){

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, *pc_in, indices);


    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);


    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

    // 刪去上三分之一部分
    int third_rows = gray.rows / 4;
    cv::Mat gray_cropped = gray(cv::Rect(0, third_rows, gray.cols, gray.rows - third_rows));

    // 做Canny邊緣檢測
    // Canny(gray_cropped, gray_cropped, 145, 105);

    // 定义变量来存储梯度
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, grad;

    // 计算x方向梯度
    // 使用高斯模糊
    cv::GaussianBlur(gray_cropped, gray_cropped , cv::Size(3, 3), 1.0);

    cv::Sobel(gray_cropped, grad_x, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // 计算y方向梯度
    cv::Sobel(gray_cropped, grad_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // 合并梯度
    cv::addWeighted(abs_grad_x, 0.25, abs_grad_y, 0.75, 0, gray_cropped);

    // 用0（黑色）填補回去
    cv::Mat gray_filled(gray.rows, gray.cols, gray.type(), cv::Scalar(0));
    gray_cropped.copyTo(gray_filled(cv::Rect(0, third_rows, gray_cropped.cols, gray_cropped.rows)));
  
  
    // cv::imshow("sobel",gray);
    // cv::waitKey(0);


    int window_size = 3; // 可以根据需要调整

    cv_bridge::CvImagePtr cv_ptr_2;
    cv_ptr_2 = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    
// processImageEdges_parallel(gray, matrix_3Dto2D, edge_first, window_size, cv_ptr_2, pc_out_edge);
 

    // #pragma omp parallel for
    for (int i = 0; i < (int) pc_in->points.size(); i++) {
        if (pc_in->points[i].x >= 0) {
            
            Eigen::Vector4d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z, 1);
            Eigen::Vector3d curr_point_image = matrix_3Dto2D * curr_point;

            curr_point_image.x() = curr_point_image.x() / curr_point_image.z();
            curr_point_image.y() = curr_point_image.y() / curr_point_image.z();

            // 检查投影点是否在边缘上
            int x = static_cast<int>(curr_point_image.x());
            int y = static_cast<int>(curr_point_image.y());
                
            if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
                
                int intensity_threshold = 30;
                // 獲取當前像素的強度值
                uchar current_intensity = gray.at<uchar>(y, x);

                // 根據當前像素值動態調整閾值
                int edge_threshold = (current_intensity < 96) ? (intensity_threshold / 2) : intensity_threshold;

                // if (gray.at<uchar>(y, x) > 0) {
                    // 检查周围像素是否部份是边缘 //0717
                    bool is_edge_nearby = false;
                    int half_window_size = window_size / 2;
                    for (int dy = -half_window_size; dy <= half_window_size; dy++) {
                        for (int dx = -half_window_size; dx <= half_window_size; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < gray.cols && ny >= 0 && ny < gray.rows) { //v4
                                if (gray.at<uchar>(ny, nx)  > gray.at<uchar>(y, x) + edge_threshold  ) {
                                    is_edge_nearby = true ;
                                    break;
                                }
                            }
                        }
                        if (is_edge_nearby) {
                            break;
                        }
                    }

                    if (is_edge_nearby) {
                        cv::circle(cv_ptr_2->image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
                        // #pragma omp critical
                        {
                            pc_out_edge->push_back(pc_in->points[i]);
                        }
                        // number++;
                    }
                    else{
                        surf_first->push_back(pc_in->points[i]);
                    }
                // }
            }
        }
    }
    double map_resolution = 0.3;
    downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2, map_resolution * 2);
    // downSamplingToMap(pc_out_edge, pc_out_edge);//0529
    std::cout << "after edge number = " << (int)pc_out_edge->points.size() << std::endl;
    
    // cv::imshow("after edge", cv_ptr_2->image);
    // cv::waitKey(0);

}



LaserProcessingClass::LaserProcessingClass(){
    
}

Double2d::Double2d(int id_in, double value_in){
    id = id_in;
    value =value_in;
};

PointsInfo::PointsInfo(int layer_in, double time_in){
    layer = layer_in;
    time = time_in;
};


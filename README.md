Nearest Neighbor Interpolation SLAM

Our system is divided into two main parts, with this explanation focusing on the feature extraction portion.
The odometry estimation part will be detailed later.

Feature Extraction:
  Removal of Invalid Points: Remove NaN values from the point cloud to ensure data integrity.
  
  Scan Line Segmentation: Segment the point cloud into multiple sub-point clouds based on LiDAR
  scan lines for independent processing.
  
  1.  Edge Feature Extraction:
    Apply the Sobel operator to the left camera image of the KITTI dataset for convolution processing
    (the final fill ratio is not 0.5:0.5;please refer to the internal code for details).

    Project the processed point cloud onto the image. Then, use a sliding window detection method to identify edge points
    based on the points near the projected point cloud.
    
    If a point is identified as an edge point, it is added to the edge points set. If not, it is added to surf_first for
    further processing in the next step.
    
    Since the Sobel processing is still affected by lighting conditions, a simple dual threshold method is used here 
    to set different standards for bright and dark areas.

    
  2.  Surface Feature Extraction:
    Apply Gaussian blur to the image.

    Use a sliding window detection method to identify surface points. If a point is identified as an edge point,
    it is added to the surface points set. Otherwise, it is discarded.
    
  3.  Building the Surface Feature KD-Tree and Applying Nearest Neighbor Interpolation (NNI):
    Build a KD-Tree index for pc_out_surf.

    Use the KD-Tree to find the nearest points for interpolation.
    
    Add the interpolated points to pc_out_surf.

  Finally, pass pc_out_edge and pc_out_surf to the odometry estimation part for processing.

Odometry Estimation:
  To be continued...

//中文說明 =========================================================================================================
最近相鄰內插法SLAM(Nearest neighbor interpolation_SLAM)

我們的系統分成兩個部分，這說明著重於特徵擷取的部分，後面里程估計部分有空再補。

特徵擷取 : 
  去除無效點： 移除點雲中的NaN值，確保資料的完整性。 掃描線分割： 根據光達掃描線，將點雲分為多個子點雲，以便獨立處理。 
  1.  提取邊緣資訊 : 先將kitti的左相機影像使用sobel算子來做捲積處理(最後回填比例非0.5:0.5，請看內部程式碼)，
      再將剛剛處理好的點雲做投影到圖片上，根據點雲附近的點做Silding Window檢測來判斷是否為邊緣點，
      是的話加入邊緣點，不是的話加入surf_first做下一步的判斷。
      因為sobel處裡完仍有光照問題，這邊先簡單的用雙閥值(Dual threshold)來讓明暗部分有不同標準。
  2.  提取面特徵 : 這邊先將圖片高斯模糊，再使用Silding Window檢測來判斷是否為邊緣點，是的話加入表面點，不是的話該點捨棄。
  3.  建立面特徵的kd_tree並且使用NNI(最近相鄰內插): 建立pc_out_surf的kd_tree索引，並且使用kdtree分別找出最近的點來進行內插。
      最後將內插出的點加入pc_out_surf
  將pc_out_edge和pc_out_surf給里程估計部分做處理

里程估計 : 
  待補

// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "KissICP.hpp"

#include <Eigen/Core>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

/**
 * [功能描述]：注册并配准一帧点云数据，执行KISS-ICP算法的核心流程
 * @param frame：输入的3D点云数据，每个点用Eigen::Vector3d表示
 * @param timestamps：每个点的时间戳向量，用于运动补偿和去畸变
 * @return Vector3dVectorTuple：返回元组，包含预处理后的点云和用于配准的源点云
 */
KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps) {
    // 第一步：预处理输入点云，执行运动补偿和去畸变处理
    // 使用上一帧的位姿增量(last_delta_)来估计帧内运动
    const auto &preprocessed_frame = preprocessor_.Preprocess(frame, timestamps, last_delta_);

    // 第二步：体素化处理，将点云下采样以减少计算量并提高配准精度
    // 返回两个不同分辨率的点云：source用于配准，frame_downsample用于地图更新
    const auto &[source, frame_downsample] = Voxelize(preprocessed_frame);

    // 第三步：获取自适应阈值参数sigma，用于控制ICP配准的收敛条件
    // sigma会根据历史配准误差动态调整
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // 第四步：计算ICP配准的初始位姿猜测
    // 使用上一帧位姿(last_pose_)和位姿增量(last_delta_)来预测当前帧位姿
    const auto initial_guess = last_pose_ * last_delta_;

    // 第五步：执行ICP点云配准算法
    // source: 当前帧待配准的点云
    // local_map_: 局部地图的体素哈希表
    // initial_guess: 初始位姿估计
    // 3.0 * sigma: 最大对应距离阈值，超过此距离的点对不参与配准
    // sigma: 核函数参数，用于鲁棒估计
    const auto new_pose = registration_.AlignPointsToMap(source,         // 待配准点云
                                                         local_map_,     // 目标地图
                                                         initial_guess,  // 初始位姿猜测
                                                         3.0 * sigma,    // 最大对应距离
                                                         sigma);         // 核函数参数

    // 第六步：计算模型预测与实际估计之间的偏差
    // 用于评估运动模型的准确性并更新自适应阈值
    const auto model_deviation = initial_guess.inverse() * new_pose;

    // 第七步：更新算法状态，为下一帧做准备
    adaptive_threshold_.UpdateModelDeviation(model_deviation);  // 更新自适应阈值模型
    local_map_.Update(frame_downsample, new_pose);              // 用配准后的点云更新局部地图
    last_delta_ = last_pose_.inverse() * new_pose;              // 计算并保存位姿增量
    last_pose_ = new_pose;                                      // 更新当前位姿为最新估计

    // 返回预处理后的原始点云和用于配准的源点云
    // preprocessed_frame: 去畸变后的完整点云，可用于建图等后续处理
    // source: 下采样后的配准点云，显示了实际参与配准计算的点
    return {preprocessed_frame, source};
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const auto source = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}

}  // namespace kiss_icp::pipeline

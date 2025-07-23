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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include "VoxelHashMap.hpp"
#include "VoxelUtils.hpp"

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

using Correspondences = tbb::concurrent_vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
using LinearSystem = std::pair<Eigen::Matrix6d, Eigen::Vector6d>;

namespace {
inline double square(double x) { return x * x; }

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

Correspondences DataAssociation(const std::vector<Eigen::Vector3d> &points,
                                const kiss_icp::VoxelHashMap &voxel_map,
                                const double max_correspondance_distance) {
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    Correspondences correspondences;
    correspondences.reserve(points.size());
    tbb::parallel_for(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        [&](const tbb::blocked_range<points_iterator> &r) {
            std::for_each(r.begin(), r.end(), [&](const auto &point) {
                const auto &[closest_neighbor, distance] = voxel_map.GetClosestNeighbor(point);
                if (distance < max_correspondance_distance) {
                    correspondences.emplace_back(point, closest_neighbor);
                }
            });
        });
    return correspondences;
}

LinearSystem BuildLinearSystem(const Correspondences &correspondences, const double kernel_scale) {
    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto &[source, target] = correspondence;
        const Eigen::Vector3d residual = source - target;
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
        return std::make_tuple(J_r, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto GM_weight = [&](const double &residual2) {
        return square(kernel_scale) / square(kernel_scale + residual2);
    };

    using correspondence_iterator = Correspondences::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(),
                                                    correspondences.cend()},
        // Identity
        LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<correspondence_iterator> &r, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                r.begin(), r.end(), J, sum_linear_systems, [&](const auto &correspondence) {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(correspondence);
                    const double w = GM_weight(residual.squaredNorm());
                    return LinearSystem(J_r.transpose() * w * J_r,        // JTJ
                                        J_r.transpose() * w * residual);  // JTr
                });
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        sum_linear_systems);

    return {JTJ, JTr};
}
}  // namespace

namespace kiss_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      // Only manipulate the number of threads if the user specifies something greater than 0
      max_num_threads_(max_num_threads > 0 ? max_num_threads
                                           : tbb::this_task_arena::max_concurrency()) {
    // This global variable requires static duration storage to be able to manipulate the max
    // concurrency from TBB across the entire class
    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads_));
}

/**
 * [功能描述]：执行点云到地图的ICP配准算法，估计最优位姿变换
 * @param frame：待配准的源点云，包含当前帧的3D点集合
 * @param voxel_map：目标地图的体素哈希表，作为配准的参考
 * @param initial_guess：初始位姿猜测，基于运动模型预测的变换矩阵
 * @param max_distance：最大对应距离阈值，超过此距离的点对将被忽略
 * @param kernel_scale：核函数尺度参数，用于鲁棒估计中的权重计算
 * @return Sophus::SE3d：返回优化后的SE(3)位姿变换矩阵
 */
Sophus::SE3d Registration::AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                                            const VoxelHashMap &voxel_map,
                                            const Sophus::SE3d &initial_guess,
                                            const double max_distance,
                                            const double kernel_scale) {
    // 检查地图是否为空，如果为空则直接返回初始猜测
    // 避免在空地图上进行无意义的配准计算
    if (voxel_map.Empty()) return initial_guess;

    // 公式(9)：复制源点云并应用初始变换
    // 将待配准点云变换到初始猜测位姿，作为ICP迭代的起始点
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // 初始化ICP累积变换矩阵，用于记录迭代过程中的总变换
    Sophus::SE3d T_icp = Sophus::SE3d();
    
    // ICP主迭代循环，最多执行max_num_iterations_次迭代
    for (int j = 0; j < max_num_iterations_; ++j) {
        // 公式(10)：数据关联步骤
        // 为源点云中的每个点在目标地图中找到最近邻对应点
        // 只保留距离小于max_distance的有效对应关系
        const auto correspondences = DataAssociation(source, voxel_map, max_distance);
        
        // 公式(11)：构建线性系统
        // 基于点对对应关系构建最小二乘优化的线性方程组
        // JTJ是海塞矩阵(Hessian)，JTr是梯度向量，kernel_scale用于鲁棒估计
        const auto &[JTJ, JTr] = BuildLinearSystem(correspondences, kernel_scale);
        
        // 求解线性系统得到位姿增量
        // 使用LDLT分解求解 JTJ * dx = -JTr，得到李代数空间的增量向量
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        
        // 将李代数增量转换为SE(3)群上的变换矩阵
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        
        // 公式(12)：更新源点云位姿
        // 用估计的变换增量更新源点云的位置
        TransformPoints(estimation, source);
        
        // 累积变换：将当前迭代的变换增量合并到总变换中
        T_icp = estimation * T_icp;
        
        // 收敛性检查：如果变换增量足够小，则认为已收敛，提前退出迭代
        // dx.norm()计算增量向量的二范数，小于阈值则停止优化
        if (dx.norm() < convergence_criterion_) break;
    }
    
    // 返回最终的变换结果
    // 将ICP迭代得到的相对变换与初始猜测组合，得到绝对位姿
    return T_icp * initial_guess;
}

}  // namespace kiss_icp

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <Eigen/Core>
#include <iostream>
#include <unordered_set>

using namespace Eigen;
using namespace std;

int main() {
  double PIXEL_NOISE = 1;
  // double OUTLIER_RATIO = 0.;
  bool ROBUST_KERNEL = true;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  // Solver type
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
  linearSolver = g2o::make_unique<
      g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
  // optimization algorithm
  g2o::OptimizationAlgorithmLevenberg *solver =
      new g2o::OptimizationAlgorithmLevenberg(
          g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
  optimizer.setAlgorithm(solver);

  vector<Vector3d> true_points;
  for (size_t i = 0; i < 500; ++i) {
    true_points.push_back(
        Vector3d((g2o::Sampler::uniformRand(0., 1.) - 0.5) * 3,
                 g2o::Sampler::uniformRand(0., 1.) - 0.5,
                 g2o::Sampler::uniformRand(0., 1.) + 3));
  }

  double focal_length = 1000.;
  Vector2d principal_point(320., 240.);
  g2o::CameraParameters *cam_params =
      new g2o::CameraParameters(focal_length, principal_point, 0.);
  cam_params->setId(0);
  optimizer.addParameter(cam_params);

  vector<g2o::SE3Quat, aligned_allocator<g2o::SE3Quat>> true_poses;
  int vertex_id = 0;
  for (size_t i = 0; i < 15; ++i) {
    Vector3d trans(i * 0.04 - 1., 0, 0);

    Eigen::Quaterniond q;
    q.setIdentity();
    g2o::SE3Quat pose(q, trans);
    g2o::VertexSE3Expmap *v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(vertex_id);
    if (i < 2) {
      v_se3->setFixed(true);
    }
    v_se3->setEstimate(pose);
    optimizer.addVertex(v_se3);
    true_poses.push_back(pose);
    vertex_id++;
  }

  int point_id = vertex_id;
  int point_num = 0;
  double sum_diff2 = 0;

  cout << endl;
  unordered_map<int, int> pointid_2_trueid;
  unordered_set<int> inliers;

  for (size_t i = 0; i < true_points.size(); ++i) {
    g2o::VertexSBAPointXYZ *v_p = new g2o::VertexSBAPointXYZ();
    v_p->setId(point_id);
    v_p->setMarginalized(true);
    v_p->setEstimate(true_points.at(i) +
                     Vector3d(g2o::Sampler::gaussRand(0., 1),
                              g2o::Sampler::gaussRand(0., 1),
                              g2o::Sampler::gaussRand(0., 1)));
    int num_obs = 0;
    for (size_t j = 0; j < true_poses.size(); ++j) {
      Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));
      if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
        ++num_obs;
      }
    }
    if (num_obs >= 2) {
      optimizer.addVertex(v_p);
      // bool inlier = true;
      for (size_t j = 0; j < true_poses.size(); ++j) {
        Vector2d z =
            cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));

        if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
          z += Vector2d(g2o::Sampler::gaussRand(0., PIXEL_NOISE),
                        g2o::Sampler::gaussRand(0., PIXEL_NOISE));
          g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(v_p));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertices().find(j)->second));
          e->setMeasurement(z);
          e->information() = Matrix2d::Identity();
          g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          e->setParameterId(0, 0);
          optimizer.addEdge(e);
        }
      }
      pointid_2_trueid.insert(make_pair(point_id, i));
      ++point_id;
      ++point_num;
    }
  }

  optimizer.initializeOptimization();
  optimizer.setVerbose(true);
  cout << "Performing full BA:" << endl;
  optimizer.optimize(10);
  cout << endl;
}
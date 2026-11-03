#include <Eigen/Eigenvalues>
#include <cmath>
#include <mrl/cuda_depth_buffer/pose_covariance.hpp>

namespace mrl {
namespace {

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

}  // namespace

double poseUncertaintyMarginReference(const Matrix6d& sigma,
                                      const Eigen::Vector3d& lever_arm,
                                      const double confidence) {
  const Eigen::Matrix3d position = sigma.topLeftCorner<3, 3>();
  const Eigen::Matrix3d cross = sigma.topRightCorner<3, 3>();
  const Eigen::Matrix3d orientation = sigma.bottomRightCorner<3, 3>();
  const Eigen::Matrix3d s = skewSymmetric(lever_arm);

  // J Sigma J^T with J = [I, -S], expanded and symmetrized against round off.
  const Eigen::Matrix3d cross_s = cross * s;
  Eigen::Matrix3d a = position + cross_s + cross_s.transpose() +
                      s * orientation * s.transpose();
  a = 0.5 * (a + a.transpose());

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(a);
  if (solver.info() != Eigen::Success) return 0.0;
  const double eigenvalue = solver.eigenvalues().maxCoeff();
  return confidence * std::sqrt(eigenvalue > 0.0 ? eigenvalue : 0.0);
}

}  // namespace mrl

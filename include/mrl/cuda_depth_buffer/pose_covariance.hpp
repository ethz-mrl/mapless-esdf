#ifndef MRL_CUDA_DEPTH_BUFFER_POSE_COVARIANCE_HPP
#define MRL_CUDA_DEPTH_BUFFER_POSE_COVARIANCE_HPP

#include <Eigen/Core>

namespace mrl {

/// Pose covariance, ordered [translation; rotation], expressed in the reference
/// frame and conditioned on the newest pose: it is the uncertainty of a frame
/// relative to the present, so it grows for older frames.
using Matrix6d = Eigen::Matrix<double, 6, 6>;

/// Readable reference for the uncertainty margin the query kernel applies.
///
/// A perturbation xi = (dt, dtheta) of a buffered pose displaces the query
/// point in that camera frame by J xi, with J = [I, -[r]x] and r the lever arm
/// from the camera centre to the query. The largest eigenvalue of J Sigma J^T
/// bounds the displacement variance in any direction, giving the c sigma margin
///
///     rho = c sqrt(lambda_max(J Sigma J^T)).
///
/// The kernel evaluates the same expression with a closed form 3x3 eigenvalue
/// instead of Eigen; test_pose_compensation_parity pins the two together.
double poseUncertaintyMarginReference(const Matrix6d& sigma,
                                      const Eigen::Vector3d& lever_arm,
                                      double confidence);

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_POSE_COVARIANCE_HPP

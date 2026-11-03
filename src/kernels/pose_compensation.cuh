#ifndef MRL_CUDA_DEPTH_BUFFER_KERNELS_POSE_COMPENSATION_CUH
#define MRL_CUDA_DEPTH_BUFFER_KERNELS_POSE_COMPENSATION_CUH

#include "kernels/common.cuh"

namespace mrl {

/// Obstacle inflation by the odometry uncertainty.
///
/// A perturbation xi = (dt, dtheta) of a buffered pose displaces the query in
/// that camera frame by J xi with J = [I, -[r]x], where r is the lever arm from
/// the camera centre to the query. The largest eigenvalue of J Sigma J^T bounds
/// the displacement variance in any direction, so
///
///     rho = c sqrt(lambda_max(J Sigma J^T))
///
/// is a c sigma margin. Every frame's truncation radius grows by its rho before
/// the search, and rho is subtracted from that frame's distance before merging,
/// which inflates obstacles instead of letting an uncertain frame erode them.
///

/// Largest eigenvalue of a symmetric 3x3 matrix given as
/// (a00, a01, a02, a11, a12, a22), by the closed form trigonometric solution.
/// Host and device share this so that both paths compute the same thing.
__host__ __device__ __forceinline__ double maxSymmetricEigenvalue3(
    const double a00,
    const double a01,
    const double a02,
    const double a11,
    const double a12,
    const double a22) {
  const double off_diagonal = a01 * a01 + a02 * a02 + a12 * a12;
  if (off_diagonal <= 0.0) {
    return fmax(a00, fmax(a11, a22));  // already diagonal
  }
  const double q = (a00 + a11 + a22) / 3.0;
  const double d0 = a00 - q;
  const double d1 = a11 - q;
  const double d2 = a22 - q;
  const double p =
      sqrt((d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * off_diagonal) / 6.0);
  if (!(p > 0.0)) return q;

  // det((A - q I) / p) / 2, expanded
  const double inv_p = 1.0 / p;
  const double b00 = d0 * inv_p, b11 = d1 * inv_p, b22 = d2 * inv_p;
  const double b01 = a01 * inv_p, b02 = a02 * inv_p, b12 = a12 * inv_p;
  double r =
      0.5 * (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) +
             b02 * (b01 * b12 - b11 * b02));
  r = fmin(1.0, fmax(-1.0, r));
  return q + 2.0 * p * cos(acos(r) / 3.0);
}

/// rho for one (frame, query) pair. `sigma` is the 6x6 covariance in row major
/// order, ordered [translation; rotation]; `r` is the lever arm in the
/// reference frame.
__host__ __device__ __forceinline__ double poseUncertaintyMargin(
    const double* sigma,
    const double rx,
    const double ry,
    const double rz,
    const double confidence) {
  // A = P + C S + (C S)^T + S Theta S^T with S = [r]x, symmetrized.
  // S = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
  const double s[9] = {0.0, -rz, ry, rz, 0.0, -rx, -ry, rx, 0.0};

  double cs[9];  // C S
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += sigma[i * 6 + 3 + k] * s[k * 3 + j];
      }
      cs[i * 3 + j] = sum;
    }
  }

  double ts[9];  // Theta S
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += sigma[(3 + i) * 6 + 3 + k] * s[k * 3 + j];
      }
      ts[i * 3 + j] = sum;
    }
  }

  // S Theta S^T. S is skew symmetric, so S^T = -S and S Theta S^T = -S (Theta
  // S).
  double sts[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += s[i * 3 + k] * ts[k * 3 + j];
      }
      sts[i * 3 + j] = -sum;
    }
  }

  double a[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      a[i * 3 + j] =
          sigma[i * 6 + j] + cs[i * 3 + j] + cs[j * 3 + i] + sts[i * 3 + j];
    }
  }
  const double eig = maxSymmetricEigenvalue3(a[0],
                                             0.5 * (a[1] + a[3]),
                                             0.5 * (a[2] + a[6]),
                                             a[4],
                                             0.5 * (a[5] + a[7]),
                                             a[8]);
  return confidence * sqrt(eig > 0.0 ? eig : 0.0);
}

/// One rho per (frame, query), plus the per pair truncation radius the search
/// runs with. Frames without a pose or without a covariance get rho = 0.
__global__ void poseCompensationKernel(
    const double* __restrict__ d_covariances,  // F x 36, row major
    const uint8_t* __restrict__ d_has_covariance,
    const float* __restrict__ d_poses,
    const float* __restrict__ d_queries_reference,
    size_t num_queries,
    size_t num_frames,
    float confidence,
    float truncation_radius,
    float* __restrict__ d_pose_compensation,
    float* __restrict__ d_frame_query_truncation) {
  const size_t i_frame_query =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i_frame_query >= num_queries * num_frames) return;
  const size_t i_frame = i_frame_query / num_queries;
  const size_t i_query = i_frame_query % num_queries;

  double rho = 0.0;
  const float* pose = d_poses + i_frame * 7;
  if (d_has_covariance[i_frame] != 0 && isPoseValid(pose)) {
    const double rx =
        static_cast<double>(d_queries_reference[i_query * 3 + 0]) -
        static_cast<double>(pose[0]);
    const double ry =
        static_cast<double>(d_queries_reference[i_query * 3 + 1]) -
        static_cast<double>(pose[1]);
    const double rz =
        static_cast<double>(d_queries_reference[i_query * 3 + 2]) -
        static_cast<double>(pose[2]);
    rho = poseUncertaintyMargin(d_covariances + i_frame * 36,
                                rx,
                                ry,
                                rz,
                                static_cast<double>(confidence));
  }

  d_pose_compensation[i_frame_query] = static_cast<float>(rho);
  if (d_frame_query_truncation) {
    d_frame_query_truncation[i_frame_query] =
        truncation_radius + static_cast<float>(rho);
  }
}

}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_KERNELS_POSE_COMPENSATION_CUH

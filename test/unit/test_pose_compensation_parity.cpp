// The uncertainty margin exists twice: poseUncertaintyMarginReference on the
// host (Eigen, readable) and the closed form in kernels/pose_compensation.cuh
// that the query kernel runs. They must not drift apart, so this test pins them
// to each other over random covariances and lever arms.
//
// The file is compiled as CUDA so that the same header the kernel uses can be
// evaluated on the host; the __host__ __device__ functions in it are ordinary
// host code here.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <mrl/cuda_depth_buffer/pose_covariance.hpp>
#include <random>
#include <vector>

#include "kernels/pose_compensation.cuh"

namespace {

/// Random symmetric positive semi-definite 6x6, from M M^T.
mrl::Matrix6d randomSigma(std::mt19937& rng, double scale) {
  std::normal_distribution<double> normal(0.0, scale);
  mrl::Matrix6d m;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) m(i, j) = normal(rng);
  }
  return m * m.transpose();
}

std::vector<double> rowMajor(const mrl::Matrix6d& sigma) {
  std::vector<double> flat(36);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) flat[i * 6 + j] = sigma(i, j);
  }
  return flat;
}

}  // namespace

TEST(PoseCompensationParityTest, MaxEigenvalueMatchesEigen) {
  std::mt19937 rng(4242);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int trial = 0; trial < 2000; ++trial) {
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) m(i, j) = normal(rng);
    }
    const Eigen::Matrix3d a = 0.5 * (m + m.transpose());

    const double closed_form = mrl::maxSymmetricEigenvalue3(a(0, 0),
                                                            a(0, 1),
                                                            a(0, 2),
                                                            a(1, 1),
                                                            a(1, 2),
                                                            a(2, 2));
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(a);
    ASSERT_EQ(solver.info(), Eigen::Success);
    const double reference = solver.eigenvalues().maxCoeff();

    EXPECT_NEAR(closed_form,
                reference,
                1e-9 * std::max(1.0, std::abs(reference)))
        << "trial " << trial;
  }
}

TEST(PoseCompensationParityTest, DiagonalAndDegenerateMatricesAgree) {
  // A diagonal matrix takes the early exit, and repeated eigenvalues push the
  // trigonometric branch to the ends of its acos domain.
  const std::vector<std::array<double, 6>> cases = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},          // zero
      {1.0, 0.0, 0.0, 1.0, 0.0, 1.0},          // identity, triple root
      {2.0, 0.0, 0.0, 2.0, 0.0, 5.0},          // double root
      {-3.0, 0.0, 0.0, -1.0, 0.0, -2.0},       // negative definite, diagonal
      {1e-14, 1e-15, 0.0, 1e-14, 0.0, 1e-14},  // tiny
  };
  for (const auto& c : cases) {
    Eigen::Matrix3d a;
    a << c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5];
    const double closed_form =
        mrl::maxSymmetricEigenvalue3(c[0], c[1], c[2], c[3], c[4], c[5]);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(a);
    EXPECT_NEAR(
        closed_form,
        solver.eigenvalues().maxCoeff(),
        1e-12 * std::max(1.0, std::abs(solver.eigenvalues().maxCoeff())));
  }
}

TEST(PoseCompensationParityTest, MarginMatchesTheEigenReference) {
  std::mt19937 rng(7);
  std::normal_distribution<double> arm(0.0, 3.0);
  for (int trial = 0; trial < 500; ++trial) {
    const mrl::Matrix6d sigma = randomSigma(rng, 0.3);
    const std::vector<double> flat = rowMajor(sigma);
    const Eigen::Vector3d r(arm(rng), arm(rng), arm(rng));
    const double confidence = 3.0;

    const double reference =
        mrl::poseUncertaintyMarginReference(sigma, r, confidence);
    const double device_math = mrl::poseUncertaintyMargin(flat.data(),
                                                          r.x(),
                                                          r.y(),
                                                          r.z(),
                                                          confidence);

    EXPECT_NEAR(device_math, reference, 1e-9 * std::max(1.0, reference))
        << "trial " << trial;
  }
}

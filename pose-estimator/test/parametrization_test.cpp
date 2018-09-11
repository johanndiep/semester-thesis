#include "gtest/gtest.h"
#include "posest/executionparametrization.h"
#include <cmath>

using posest::ExecutionResults;
using mrpt::math::CQuaternionDouble;

TEST(Parametrization, execution_results_position) {
    CPose3DQuat p1(1, 1, 1, CQuaternionDouble(1, 0, 0, 0)); // pose offset without rotation
    CPose3DQuat p2(0, 0, 0, CQuaternionDouble(1, 0, 0, 0));
    ceres::Solver::Summary summary;
    ExecutionResults results(p2, p1, summary);

    EXPECT_EQ(0, results.get_angular_error());
    EXPECT_EQ(std::sqrt(3), results.get_distance_error());
    EXPECT_EQ(-1, results.get_position_error()[0]);
    EXPECT_EQ(-1, results.get_position_error()[1]);
    EXPECT_EQ(-1, results.get_position_error()[2]);
}

TEST(Parametrization, execution_results_rotation) {
    CPose3DQuat p1(0, 0, 0, CQuaternionDouble(1, 0, 0, 0)); // no rotation
    CPose3DQuat p2(0, 0, 0,
                   CQuaternionDouble(std::sqrt(2) / 2, 0, 0, std::sqrt(2) / 2)); // 90deg rotation around z axis
    ceres::Solver::Summary summary;
    ExecutionResults results(p2, p1, summary);

    EXPECT_NEAR(M_PI_2, results.get_angular_error(), 1e-8);
    EXPECT_EQ(0, results.get_distance_error());
}

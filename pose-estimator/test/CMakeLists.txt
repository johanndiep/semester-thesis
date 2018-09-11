include(GoogleTest)

# add all test c++ files to the following list
add_executable(pose_estimator_tests
        test_all.cpp
        blurrer_test.cpp
        solver_test.cpp
        reader_test.cpp reprojector_test.cpp parametrization_test.cpp)
target_include_directories(pose_estimator_tests PUBLIC ${GTEST_INCLUDE_DIR})
target_link_libraries(pose_estimator_tests PUBLIC libgtest posest ${MRPT_LIBRARIES})
add_dependencies(pose_estimator_tests libgtest)

gtest_discover_tests(pose_estimator_tests)
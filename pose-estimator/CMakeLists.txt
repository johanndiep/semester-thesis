cmake_minimum_required(VERSION 3.10)
project(pose_estimator)

set(CMAKE_CXX_STANDARD 11)

##
# search for dependency librararies
##

# ceres-solver
find_package(Ceres)
include_directories(${CERES_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# mrpt
set(MRPT_DONT_USE_DBG_LIBS true) # this for debug
find_package(MRPT REQUIRED base)
include_directories(${MRPT_INCLUDE_DIRS})

##
# Prepare testing framework google test
##
include(ExternalProject)
ExternalProject_Add(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.8.0
        INSTALL_COMMAND "")

ExternalProject_Get_Property(googletest source_dir binary_dir)
# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest INTERFACE)
add_dependencies(libgtest gtest)
target_link_libraries(libgtest INTERFACE "${binary_dir}/googlemock/libgmock.a" "${binary_dir}/googlemock/gtest/libgtest.a")
target_include_directories(libgtest INTERFACE "${source_dir}/googletest/include" "${source_dir}/googlegmock/include")

enable_testing()
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(cmd)

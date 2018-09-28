# add all c++ files which are part of posest to this list
add_library(posest STATIC
        posest/solver.cpp
        posest/reprojector.cpp
        posest/blurrer.cpp
        posest/dataset.cpp
        posest/executionparametrization.h
        posest/posest.h)

target_include_directories(posest PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(posest ${CERES_LIBRARIES} ${OpenCV_LIBS} ${MRPT_LIBRARIES})

# main executable
add_executable(pose-estimator pose-estimator.cpp)
target_link_libraries(pose-estimator posest)

# create create a 3D point cloud file
add_executable(3d-pointcloud 3d-pointcloud.cpp)
target_link_libraries(3d-pointcloud posest)

# warp an image into another camera pose
add_executable(reproject reproject.cpp)
target_link_libraries(reproject posest)

# create a blurred image from a ref image and a target image pose
add_executable(blur blur.cpp)
target_link_libraries(blur posest)

# main executable for pyramid style 
add_executable(coarse-to-sparse coarse-to-sparse.cpp)
target_link_libraries(coarse-to-sparse posest)
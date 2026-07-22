cmake_minimum_required(VERSION 3.16)
project(robotcar_utils_cpp)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
# Gazebo Fortress marker service (draws path in the 3D scene)
find_package(ignition-transport11 REQUIRED)
find_package(ignition-msgs8 REQUIRED)

add_executable(path_marker_node src/path_marker_node.cpp)
ament_target_dependencies(path_marker_node rclcpp nav_msgs)
target_link_libraries(path_marker_node
  ignition-transport11::ignition-transport11
  ignition-msgs8::ignition-msgs8
)
install(TARGETS path_marker_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

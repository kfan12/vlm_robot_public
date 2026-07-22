cmake_minimum_required(VERSION 3.16)
project(mpc_tracker_cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# osqp and osqp-eigen are installed to /usr/local (built from source)
set(OsqpEigen_DIR "/usr/local/lib/cmake/OsqpEigen" CACHE PATH "")

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

add_library(mpc_utils
  src/vehicle_model.cpp
  src/path_utils.cpp
)
target_include_directories(mpc_utils PUBLIC include)
ament_target_dependencies(mpc_utils rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(mpc_utils Eigen3::Eigen tf2::tf2)

add_executable(fake_path_node src/fake_path_node.cpp)
ament_target_dependencies(fake_path_node rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(fake_path_node Eigen3::Eigen tf2::tf2)
install(TARGETS fake_path_node DESTINATION lib/${PROJECT_NAME})

add_executable(pure_pursuit_node
  src/pure_pursuit_node.cpp
  src/vehicle_model.cpp
  src/path_utils.cpp
)

target_include_directories(pure_pursuit_node PUBLIC include)
ament_target_dependencies(pure_pursuit_node rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs visualization_msgs)
target_link_libraries(pure_pursuit_node Eigen3::Eigen tf2::tf2)
install(TARGETS pure_pursuit_node DESTINATION lib/${PROJECT_NAME})


add_executable(mpc_tracker_node
  src/mpc_tracker_node.cpp
  src/mpc_solver.cpp
  src/path_utils.cpp
  src/vehicle_model.cpp
)

target_include_directories(mpc_tracker_node PUBLIC include)
ament_target_dependencies(mpc_tracker_node rclcpp nav_msgs geometry_msgs std_msgs ackermann_msgs tf2 tf2_ros tf2_geometry_msgs visualization_msgs)
target_link_libraries(mpc_tracker_node Eigen3::Eigen tf2::tf2 osqp::osqp OsqpEigen::OsqpEigen)
install(TARGETS mpc_tracker_node DESTINATION lib/${PROJECT_NAME})



add_executable(path_relay_node src/path_relay_node.cpp)
ament_target_dependencies(path_relay_node rclcpp nav_msgs geometry_msgs)
install(TARGETS path_relay_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

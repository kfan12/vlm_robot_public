cmake_minimum_required(VERSION 3.16)
project(robotcar_description)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf launch config meshes
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
ament_package()

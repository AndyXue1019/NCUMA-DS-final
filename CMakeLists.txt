cmake_minimum_required(VERSION 3.0.2)
project(imu_mouse)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  tf
)


catkin_package(
  rospy
  sensor_msgs
  geometry_msgs
  tf
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


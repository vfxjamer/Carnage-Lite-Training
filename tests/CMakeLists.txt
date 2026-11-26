cmake_minimum_required(VERSION 3.18)
project(CarnageScheduleTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_executable(CarnageScheduleTests TrainingScheduleTests.cpp)
target_include_directories(CarnageScheduleTests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src")
add_test(NAME CarnageScheduleTests COMMAND CarnageScheduleTests)

add_executable(CarnageRewardMetricsTests RewardEpisodeMetricsTests.cpp)
target_include_directories(CarnageRewardMetricsTests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src")
add_test(NAME CarnageRewardMetricsTests COMMAND CarnageRewardMetricsTests)

cmake_minimum_required(VERSION 3.10)  

project(SNO)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(EXISTS ${CMAKE_SOURCE_DIR}/ignore)
    message(STATUS "find ignore folder")
    include(ignore/path.cmake)
else()
    message(STATUS "no ignore folder")
endif()

# 添加UDOBoolean目录
add_subdirectory(sno)
add_subdirectory(main1)

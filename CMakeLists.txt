cmake_minimum_required(VERSION 3.10)  

# 现代CMake使用以下方式查找CUDA
# 设置CUDA作为启用的语言
enable_language(CUDA)

# 设置CUDA编译模式
option(CUDA_SEPARABLE_COMPILATION "Enable separable compilation" ON)

project(SNO CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置最低CUDA版本要求
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 确保默认CUDA分离编译在所有目标中一致
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 输出CUDA版本信息
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

if(EXISTS ${CMAKE_SOURCE_DIR}/ignore)
    message(STATUS "find ignore folder")
    include(ignore/path.cmake)
else()
    message(STATUS "no ignore folder")
endif()

# 添加UDOBoolean目录
add_subdirectory(sno)
add_subdirectory(main1)
add_subdirectory(main_inner_init)

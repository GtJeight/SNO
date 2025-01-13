# 尝试查找Eigen3
find_package(Eigen3 REQUIRED)

if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please specify the path to Eigen3 manually.")
endif()

find_package(MKL REQUIRED)

if(NOT MKL_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please specify the path to Eigen3 manually.")
endif()

add_subdirectory(ANN)
add_subdirectory(HLBFGS)

# 添加源文件
file(GLOB_RECURSE sno_src *.c *.cpp *.h *.hpp)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${sno_src})

# 创建目标  
add_library(sno ${sno_src})

target_link_libraries(sno PUBLIC ANN)
target_link_libraries(sno PUBLIC HLBFGS)
target_link_libraries(sno PUBLIC MKL)

#target_link_libraries(InverseSubdiv PUBLIC v2)

target_include_directories(InverseSubdiv PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(InverseSubdiv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(InverseSubdiv PUBLIC ${EIGEN3_INCLUDE_DIR})
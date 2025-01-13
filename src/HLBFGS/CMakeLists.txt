# 添加源文件
file(GLOB_RECURSE HLBFGS_src *.c *.cpp *.h *.hpp)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HLBFGS_src})

# 创建目标  
add_library(HLBFGS ${HLBFGS_src})

target_include_directories(HLBFGS PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(HLBFGS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
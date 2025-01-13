# 添加源文件
file(GLOB_RECURSE inversesubdiv_src *.c *.cpp *.h *.hpp)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ANN_src})

# 创建目标  
add_library(ANN ${ANN_src})

target_include_directories(ANN PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(ANN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
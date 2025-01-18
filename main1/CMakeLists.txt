cmake_minimum_required(VERSION 3.10)  

project(main1)

file(COPY ${OMP_BIN_DIR}/libiomp5md.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
file(COPY ${OMP_BIN_DIR}/libiomp5md.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)

# 添加源文
file(GLOB_RECURSE main1_src ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# 创建目标  
add_executable(main1 ${main1_src})

target_link_libraries(main1 PUBLIC sno)

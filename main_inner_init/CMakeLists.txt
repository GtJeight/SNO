cmake_minimum_required(VERSION 3.10)  

project(main_inner_init)

file(COPY ${OMP_BIN_DIR}/libiomp5md.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
file(COPY ${OMP_BIN_DIR}/libiomp5md.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)

# 添加源文
file(GLOB_RECURSE main_inner_init_src ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# 创建目标  
add_executable(main_inner_init ${main_inner_init_src})

target_link_libraries(main_inner_init PUBLIC sno)

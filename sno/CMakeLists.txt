# 尝试查找Eigen3
find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please specify the path to Eigen3 manually.")
endif()

# 配置MKL
set(MKL_LINK static)
set(MKL_INTERFACE_FULL intel_lp64)
message(${MKL_LINK})
message(${MKL_INTERFACE_FULL})

find_package(MKL REQUIRED)
if(NOT MKL_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please specify the path to Eigen3 manually.")
endif()

add_definitions(-DEIGEN_USE_MKL_ALL)

link_directories(MKL_LIB_DIR)
link_directories(OMP_LIB_DIR)

add_subdirectory(ANN)
add_subdirectory(HLBFGS)

# 添加源文件
file(GLOB sno_src *.c *.cpp *.h *.hpp)
file(GLOB sno_cuda_src *.cu)

list(REMOVE_ITEM sno_src ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${sno_src} ${sno_cuda_src})

# 创建目标  
add_library(sno ${sno_src} ${sno_cuda_src})

# 设置CUDA架构（根据你的RTX 3070 laptop显卡计算能力）
set_target_properties(sno PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# 启用CUDA全局选项和设备代码链接
set_property(TARGET sno PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET sno PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(sno PUBLIC ANN)
target_link_libraries(sno PUBLIC HLBFGS)
target_link_libraries(sno PUBLIC Eigen3::Eigen)
target_link_libraries(
    sno
    PUBLIC 
    ${mkl_scalapack_lp64_file}
    ${mkl_cdft_core_file}
    ${mkl_intel_lp64_file}
    ${mkl_intel_thread_file}
    ${mkl_core_file}
    ${mkl_blacs_intelmpi_lp64_file}
    ${OMP_LIBRARY}
    ${IMPI_LIBRARY}
    )

#target_link_libraries(InverseSubdiv PUBLIC v2)

target_include_directories(sno PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(sno PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(sno PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(sno PUBLIC ${MKL_INCLUDE})
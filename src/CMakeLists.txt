# 收集当前目录下所有 .h 和 .cpp 文件
FILE(GLOB ucoslam_hdrs_base "*.h" )
FILE(GLOB ucoslam_srcs_base "*.cpp")
# add_definitions(-DXFEATURES2D)      # 06.21 启用 XFeatures2D 特征提取器支持

# add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY)  # 06.23 解决 Eigen3 编译时的对齐问题
add_compile_definitions(
  EIGEN_DONT_ALIGN_STATICALLY
  EIGEN_DONT_VECTORIZE
)  # 06.23 解决 Eigen3 编译时的对齐问题

# 明确列出特征提取器相关的头文件和源文件
SET(ucoslam_fextractors_hdr featureextractors/ORBextractor.h featureextractors/feature2dserializable.h )
SET(ucoslam_fextractors_src featureextractors/ORBextractor.cpp featureextractors/feature2dserializable.cpp)

# find_package(OpenCV 4.5.5 REQUIRED)         # 查找 OpenCV 4.5.5 版本 06.20

# OpenCV版本高于3时才加入 gridextractor 模块
# 条件编译，如果 OpenCV 大于3，加入额外的 grid 特征提取器。
IF(${OpenCV_VERSION} STRGREATER 3)
    SET(ucoslam_fextractors_hdr ${ucoslam_fextractors_hdr} featureextractors/gridextractor.h )
    SET(ucoslam_fextractors_src ${ucoslam_fextractors_src} featureextractors/gridextractor.cpp )
ENDIF()

# 自动收集其他模块的源代码和头文件
# 这些是工程的其他子模块代码，例如优化器、地图类型、基础类型、工具函数等
FILE(GLOB ucoslam_opt_src "optimization/*.cpp" )
FILE(GLOB ucoslam_opt_hdr "optimization/*.h" )
FILE(GLOB ucoslam_map_hdr "map_types/*.h" )
FILE(GLOB ucoslam_map_src "map_types/*.cpp" )
FILE(GLOB ucoslam_utils_hdr "utils/*.h" )
FILE(GLOB ucoslam_utils_src "utils/*.cpp" )
FILE(GLOB ucoslam_basictypes_hdr "basictypes/*.h" )
FILE(GLOB ucoslam_basictypes_src "basictypes/*.cpp" )



# 将所有收集到的源文件编译成一个库（静态或动态取决于全局设置）
ADD_LIBRARY(${PROJECT_NAME}  
            ${ucoslam_hdrs_base} 
            ${ucoslam_srcs_base} 
            ${ucoslam_fextractors_src} 
            ${ucoslam_fextractors_hdr}  
            ${ucoslam_basictypes_src} 
            ${ucoslam_basictypes_hdr}  
            ${ucoslam_map_hdr} 
            ${ucoslam_map_src} 
            ${ucoslam_utils_hdr} 
            ${ucoslam_utils_src} 
            ${ucoslam_opt_src} 
            ${ucoslam_opt_hdr})

# 设置库 25.06.10
target_include_directories(ucoslam PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../utils>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../basictypes>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../featureextractors>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../map_types>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../optimization>
  # 添加本地安装的 opencv_xfeatures2d 头文件路径
  /home/kilox/cloud_mapping/src/ucoslam2/3rdparty/comp/local_install/opencv_xfeatures2d/include
)


# 设置目标库的属性 设置版本、符号、输出名等编译细节，使得结果可控、可复用
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES 				  # create *nix style library versions + symbolic links
    DEFINE_SYMBOL UCOSLAM_DSO_EXPORTS       # 预处理符号定义
    VERSION ${PROJECT_VERSION}              # 版本号用于库版本控制
    SOVERSION ${PROJECT_SOVERSION}          # 兼容版本号
    CLEAN_DIRECT_OUTPUT 1 					# 清理直接输出          # allow creating static and shared libs without conflicts
    OUTPUT_NAME "${PROJECT_NAME}${PROJECT_DLLVERSION}" 	          # 控制库的输出名称（加版本号后缀）# avoid conflicts between library and binary target names
    )

# 06.21 添加本地构建的 opencv_xfeatures2d 库所在目录
link_directories(/home/kilox/cloud_mapping/src/ucoslam2/3rdparty/comp/local_install/opencv_xfeatures2d/lib)

# 链接依赖库（OpenCV、Eigen等），这些库在 dependencies.cmake 中配置
TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
        ${UCOSLAM_REQUIRED_LIBRARIES} 
        # opencv_core
        # opencv_features2d
        # opencv_highgui
        # opencv_imgcodecs
        opencv_xfeatures2d
        )
# TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})       # 链接 OpenCV 库 06.20

# 安装规则（install）部分
# 安装目标库文件（dll/so/a）到指定目录
INSTALL(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
    LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
    ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory


# 安装头文件部分 安装主目录下的头文件到 include/ucoslam/
INSTALL(FILES ${ucoslam_hdrs_base}
    DESTINATION include/${PROJECT_NAME}
    COMPONENT main)

#INSTALL(FILES ${ucoslam_opt_hdr}   DESTINATION include/${PROJECT_NAME}/optimizers   COMPONENT main)
#INSTALL(FILES ${ucoslam_fextractors_hdr} DESTINATION include/${PROJECT_NAME}/featureextractors COMPONENT main)
#INSTALL(FILES ${ucoslam_utils_hdr}       DESTINATION include/${PROJECT_NAME}/utils COMPONENT main)

# 安装 map_types 与 basictypes 的头文件到对应的子目录下
INSTALL(FILES ${ucoslam_map_hdr}         DESTINATION include/${PROJECT_NAME}/map_types COMPONENT main)
INSTALL(FILES ${ucoslam_basictypes_hdr}         DESTINATION include/${PROJECT_NAME}/basictypes COMPONENT main)


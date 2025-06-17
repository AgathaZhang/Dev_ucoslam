# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.5) # 最低CMake版本要求


project(ucoslam VERSION "1.1.1" LANGUAGES CXX)  # 定义项目名称、版本和使用的语言
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")  # 定义项目的SOVERSION（共享库版本控制）
set(CMAKE_INCLUDE_CURRENT_DIR ON)   # 使用当前目录作为头文件默认包含路径（用于 include_directories）
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 要求强制启用C++标准
set(CMAKE_CXX_EXTENSIONS ON)        # 启用C++扩展，如 gnu++17，而非纯标准 with compiler extensions like gnu++11
set(CMAKE_CXX_STANDARD 17)          # 设置使用的C++标准为 C++17  

# 设置编译选项，-g 用于生成调试信息 25.06.09
add_compile_options(-g)

IF(NOT CMAKE_BUILD_TYPE )
  #  SET( CMAKE_BUILD_TYPE "Release" )
# else()
   SET( CMAKE_BUILD_TYPE "Debug" )
   message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
ENDIF()

#------------------------------------------------------
# Lib Names and Dirs
#------------------------------------------------------
# 包含 CMake 子模块：选项定义（如 BUILD_UTILS）、编译器设置和依赖配置
include(cmake/options.cmake)
include(cmake/compiler.cmake)
include(cmake/dependencies.cmake)

# 针对 Windows 系统的构建配置
if(WIN32)
    # 定义 DLL 版本号后缀Postfix of DLLs:
    SET(PROJECT_DLLVERSION "${PROJECT_VERSION_MAJOR}${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}")
    # 设置 Windows 平台下的输出路径（执行文件、库等）
    SET(RUNTIME_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls and binaries")
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for binaries")
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls")
else()
    # Unix/Linux下不加后缀 Postfix of so's:
    set(PROJECT_DLLVERSION) 
endif()



# 添加子目录 src，会在 src/CMakeLists.txt 中定义源文件编译逻辑
ADD_SUBDIRECTORY(src)

# 如果启用构建 utils 工具目录，则添加构建
IF (BUILD_UTILS)
ADD_SUBDIRECTORY(utils)
ENDIF()

# RGBD版本的工具模块
IF(BUILD_UTILS_RGBD)
ADD_SUBDIRECTORY(utils_rgbd)
ENDIF()

# 开发者调试用的工具模块
IF (BUILD_UTILS_DEVELOPER)
  ADD_SUBDIRECTORY(utils_developer)
ENDIF()

# 如果启用测试模块，则添加测试代码构建
IF (BUILD_TESTS)
ADD_SUBDIRECTORY(tests)
#ADD_SUBDIRECTORY(old_tests)
ENDIF()

# 如果启用调试测试，构建 debugtests 模块
IF(BUILD_DEBUGTESTS)
ADD_SUBDIRECTORY(debugtests)
ENDIF()


# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
# 用于生成 uninstall 脚本
CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
# 定义 uninstall 构建目标，可通过 make uninstall 删除安装文件
ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")


# ----------------------------------------------------------------------------
# create configuration file from .in file (If you use windows take care with paths)
# ----------------------------------------------------------------------------
# 将模板文件 config.cmake.in 配置成实际的配置文件
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/cmake/config.cmake.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
# 安装该配置文件到 CMake 查找路径中，供其他项目 find_package 使用
INSTALL(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION share/${PROJECT_NAME} )

# 显示构建信息（如版本号、编译选项等）
include(cmake/display.cmake)


# ----------------------------------------------------------------------------
#   Package Generator (如 .deb 安装包)
# ----------------------------------------------------------------------------
# 如果启用构建 Debian 安装包，包含打包配置
IF(BUILD_DEBPACKAGE)
include(cmake/cpack.cmake)
ENDIF()

# ----------------------------------------------------------------------------
#   通过 add_subdirectory 或 find_package(ucoslam) 使用
# ----------------------------------------------------------------------------
# 安装构建后的库
install(TARGETS ucoslam
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
# 安装头文件到 include 目录，供其他项目使用
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h")

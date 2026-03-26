cmake_minimum_required(VERSION 3.5)
project(BA_tracker)

# ========================================================================== #
#                      COMPILATION FLAGS CONFIG                              #
# ========================================================================== #
# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^MSVC$")
  # under windows, explicitly enable math defines like M_PI
  # allow larget objects than default
  # enable parallel compilation
  add_definitions("-D _USE_MATH_DEFINES /MP /fp:fast /INCREMENTAL")
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# =========================================================================== #
#                                 Find Packages                               #
# =========================================================================== #
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# ========================================================================== #
#                        Add package include/lib path                        # 
# ========================================================================== #
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/fmt)
include_directories(${PROJECT_SOURCE_DIR}/third_party/Sophus)
include_directories(${PROJECT_SOURCE_DIR})

if(CUDA_FOUND)
  MESSAGE(STATUS "Found CUDA verision: " ${CUDA_VERSION_STRING} " installed in: " ${CUDA_TOOLKIT_ROOT_DIR})
  include_directories(${CUDA_INCLUDE_DIRS})
  add_definitions(-DCOMPILE_WITH_CUDA)
endif()

# ========================================================================== #
#                             Build libs and targets                         # 
# ========================================================================== #
add_subdirectory(BA_tracker)
add_subdirectory(test)

# ========================================================================== #
#                                   Install                                  # 
# ========================================================================== #
MESSAGE("-- Installation Dir: " ${CMAKE_INSTALL_PREFIX})
MESSAGE("-- Binary Installation Dir: " ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
MESSAGE("-- Header Installation Dir: " ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
MESSAGE("-- Library Installation Dir: " ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/BA_tracker"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN "*.hpp"
        PATTERN "*.inc"
)

file(GLOB FILES_CORE_HEADERS CONFIGURE_DEPENDS
        "*.h"
        "*.hpp"
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/feature_detectors/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/image_proc/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/kd_tree/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/measurements/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sensors/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/states/*.h
)

file(GLOB FILES_CORE_SRCS CONFIGURE_DEPENDS
        "*.cpp"
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/feature_detectors/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_proc/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kd_tree/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/measurements/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sensors/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/states/*.cpp
)

set(FILES_CORE_ALL
    ${FILES_CORE_HEADERS}
    ${FILES_CORE_SRCS}
)

add_library(slam_core SHARED ${FILES_CORE_ALL})
target_link_libraries(slam_core ${OpenCV_LIBS} fmt)

if (CUDA_FOUND)
    file(GLOB FILES_CORE_CUDA
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)
        
    cuda_add_library(core_cuda SHARED
        ${FILES_CORE_CUDA}
        OPTIONS 
        -std c++11)
        
    target_link_libraries(slam_core ${CUDA_LIBRARIES} core_cuda)

    install(TARGETS core_cuda
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(TARGETS slam_core
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

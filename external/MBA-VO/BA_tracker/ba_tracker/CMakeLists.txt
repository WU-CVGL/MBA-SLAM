file(GLOB FILES_BLUR_TRACKER_HEADERS CONFIGURE_DEPENDS
        "*.h"
        "*.hpp"
)

file(GLOB FILES_BLUR_TRACKER_SRCS CONFIGURE_DEPENDS
    "*.cpp"
)

add_library(slam_blur_tracker SHARED ${FILES_BLUR_TRACKER_SRCS})
target_link_libraries(slam_blur_tracker slam_core slam_utils)

if (CUDA_FOUND)
    file(GLOB FILES_VO_BLUR_TRACKER_CUDA
        "*.cu")
        
    cuda_add_library(cuda_blur_tracker SHARED
        ${FILES_VO_BLUR_TRACKER_CUDA}
        OPTIONS 
        -std c++11)

    target_link_libraries(cuda_blur_tracker ${CUDA_LIBRARIES} slam_core)
    set_target_properties(cuda_blur_tracker PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(slam_blur_tracker cuda_blur_tracker)
    
    install(TARGETS cuda_blur_tracker
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(TARGETS slam_blur_tracker
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

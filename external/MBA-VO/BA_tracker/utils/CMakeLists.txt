file(GLOB FILES_UTILS_HEADERS CONFIGURE_DEPENDS
        "*.h"
        "*.hpp"
)

file(GLOB FILES_UTILS_SRCS CONFIGURE_DEPENDS
        "*.cpp"
)

set(FILES_UTILS_ALL
    ${FILES_UTILS_HEADERS}
    ${FILES_UTILS_SRCS})

add_library(slam_utils SHARED ${FILES_UTILS_ALL})
target_link_libraries(slam_utils slam_core)

install(TARGETS slam_utils
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

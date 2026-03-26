set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/)

add_executable(test_blur_aware_tracker_sequential test_blur_aware_tracker_sequential.cpp)
target_link_libraries(test_blur_aware_tracker_sequential slam_core slam_blur_tracker ${OpenCV_LIBS})
target_compile_options(test_blur_aware_tracker_sequential PRIVATE -fsanitize=address)
target_link_options(test_blur_aware_tracker_sequential PRIVATE -fsanitize=address)

add_executable(test_blur_aware_tracker_sequential_replica test_blur_aware_tracker_sequential_replica.cpp)
target_link_libraries(test_blur_aware_tracker_sequential_replica slam_core slam_blur_tracker ${OpenCV_LIBS})
target_compile_options(test_blur_aware_tracker_sequential_replica PRIVATE -fsanitize=address)
target_link_options(test_blur_aware_tracker_sequential_replica PRIVATE -fsanitize=address)

add_executable(test_blur_aware_tracker_modules test_blur_aware_tracker_modules.cpp)
target_link_libraries(test_blur_aware_tracker_modules slam_core slam_blur_tracker)

install(TARGETS test_blur_aware_tracker_sequential test_blur_aware_tracker_modules
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(parseq-trace)

list(APPEND CMAKE_PREFIX_PATH "/home/anlab/hovduc/parseq-trace/lib/libtorch/share/cmake")

find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(parseq-trace main.cpp)
# Link Torch libraries
target_link_libraries(parseq-trace ${TORCH_LIBRARIES})

target_link_libraries(parseq-trace ${OpenCV_LIBS})
set_property(TARGET parseq-trace PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET parseq-trace
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:parseq-trace>)
endif (MSVC)
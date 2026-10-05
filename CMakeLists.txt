cmake_minimum_required(VERSION 3.24)
project(rmcs_rl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-packed-bitfield-compat")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/Observation.msg"
    "msg/Action.msg"
    "msg/PolicyStatus.msg"
    DEPENDENCIES std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

set(ONNXRUNTIME_VERSION 1.20.0)
# Select the ONNX Runtime build that matches the target architecture. The cross
# toolchain passes RMCS_TARGET_ARCH (arm64/amd64) into every package configure;
# fall back to the CMake system processor, then to the native x86_64 build.
if(NOT DEFINED ONNXRUNTIME_PLATFORM)
    if(RMCS_TARGET_ARCH STREQUAL "arm64")
        set(ONNXRUNTIME_PLATFORM linux-aarch64)
    elseif(RMCS_TARGET_ARCH STREQUAL "amd64")
        set(ONNXRUNTIME_PLATFORM linux-x64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(ONNXRUNTIME_PLATFORM linux-aarch64)
    else()
        set(ONNXRUNTIME_PLATFORM linux-x64)
    endif()
endif()

# SHA256 of the official release archives; keep in sync with tool/install_rl_deps.sh.
if(ONNXRUNTIME_PLATFORM STREQUAL "linux-aarch64")
    set(ONNXRUNTIME_SHA256 b4d7c6e2c45f8edabe5d28e9bc59ec8d5a4a4af36660cda16e94b2ad85f2a52a)
else()
    set(ONNXRUNTIME_SHA256 aa70d48b22e264b82e83f63245b51ddc9a47ae4a3a66903efaff1ba68b7b5930)
endif()

set(ONNXRUNTIME_ROOT_DIR ${CMAKE_BINARY_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION})
if(NOT EXISTS ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_cxx_api.h)
    set(ONNXRUNTIME_ARCHIVE ${CMAKE_BINARY_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz)
    if(NOT EXISTS ${ONNXRUNTIME_ARCHIVE})
        file(DOWNLOAD
            https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz
            ${ONNXRUNTIME_ARCHIVE}
            EXPECTED_HASH SHA256=${ONNXRUNTIME_SHA256}
            SHOW_PROGRESS
        )
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RESULT_VARIABLE ONNXRUNTIME_EXTRACT_RESULT
    )
    if(NOT ONNXRUNTIME_EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ONNX Runtime archive: ${ONNXRUNTIME_ARCHIVE}")
    endif()
endif()

ament_auto_add_library(rmcs_rl_bridge SHARED
    src/rl_bridge.cpp
    src/rl_bridge/config.cpp
    src/rl_bridge/observation_history.cpp
    src/rl_bridge/utility.cpp
    src/parameters.cpp
    src/rl_bridge/joint_config.cpp
    src/rl_bridge/term_parser.cpp
    src/rl_bridge/interface_binding.cpp
    src/rl_bridge/observation.cpp
    src/rl_bridge/action_channel.cpp
    src/policy_server_launcher.cpp
)
target_include_directories(rmcs_rl_bridge PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/rmcs_rl>
)
target_link_libraries(rmcs_rl_bridge ${cpp_typesupport_target})

ament_auto_add_executable(policy_server
    src/policy_server.cpp
    src/policy/policy_model.cpp
    src/policy/onnx_runtime.cpp
    src/parameters.cpp
)
target_include_directories(policy_server PRIVATE
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(policy_server ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
target_link_libraries(policy_server ${cpp_typesupport_target})
install(TARGETS policy_server DESTINATION lib/${PROJECT_NAME})

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

install(DIRECTORY models DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(runtime_test
        test/runtime_test.cpp
        src/policy/policy_model.cpp
        src/policy/onnx_runtime.cpp
    )
    target_include_directories(runtime_test PRIVATE src ${ONNXRUNTIME_ROOT_DIR}/include)
    target_link_libraries(runtime_test rmcs_rl_bridge ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
    target_compile_definitions(runtime_test PRIVATE
        RMCS_RL_IDENTITY_FIXTURE="${PROJECT_SOURCE_DIR}/test/data/normalized_identity.onnx"
        RMCS_RL_DUAL_FIXTURE="${PROJECT_SOURCE_DIR}/test/data/dual_history.onnx")
endif()

ament_export_dependencies(rosidl_default_runtime)

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)

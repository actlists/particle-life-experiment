# SDL3 CMake version configuration file:
# This file is meant to be placed in a cmake subfolder of SDL3-devel-3.2.18-VC.zip

set(PACKAGE_VERSION "3.2.18")

if(PACKAGE_FIND_VERSION_RANGE)
    # Package version must be in the requested version range
    if ((PACKAGE_FIND_VERSION_RANGE_MIN STREQUAL "INCLUDE" AND PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION_MIN)
        OR ((PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL "INCLUDE" AND PACKAGE_VERSION VERSION_GREATER PACKAGE_FIND_VERSION_MAX)
        OR (PACKAGE_FIND_VERSION_RANGE_MAX STREQUAL "EXCLUDE" AND PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION_MAX)))
        set(PACKAGE_VERSION_COMPATIBLE FALSE)
    else()
        set(PACKAGE_VERSION_COMPATIBLE TRUE)
    endif()
else()
    if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
        set(PACKAGE_VERSION_COMPATIBLE FALSE)
    else()
        set(PACKAGE_VERSION_COMPATIBLE TRUE)
        if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
            set(PACKAGE_VERSION_EXACT TRUE)
        endif()
    endif()
endif()

# if the using project doesn't have CMAKE_SIZEOF_VOID_P set, fail.
if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "")
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/sdlcpu.cmake")
SDL_DetectTargetCPUArchitectures(_detected_archs)

# check that the installed version has a compatible architecture as the one which is currently searching:
if(NOT(SDL_CPU_X86 OR SDL_CPU_X64 OR SDL_CPU_ARM64 OR SDL_CPU_ARM64EC))
    set(PACKAGE_VERSION "${PACKAGE_VERSION} (X86,X64,ARM64)")
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()

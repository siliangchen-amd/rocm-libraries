// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Backing logic for __asan_default_suppressions(), which src/AsanDefaultSuppressions.cpp defines.
//
// It lives in a header so it is reachable from a test translation unit, and is deliberately not
// guarded by ADDRESS_SANITIZER so the tests run in every configuration rather than only under ASAN.
//
// Everything here runs inside the ASan runtime's start-up, before InitializeAsanInterceptors(), so
// it must not call libc. An intercepted function's real-function pointer is still null that early
// and calling one jumps to address 0. That rules out getenv(), strlen(), strcmp() and memcmp(), so
// the environment is read through raw syscalls and compared by hand.
namespace hipdnn_test_sdk::utilities::asan
{

// Upstream rocBLAS/Tensile data race on the lazy placeholder-library load: a solution matching
// table is read while an std::async loader thread deserializes into it and reallocates the backing
// storage. AIBTINFRA-48, ROCm/rocm-libraries#8869.
inline constexpr const char* K_DEFAULT_SUPPRESSIONS = "interceptor_via_fun:*findBestKeyMatch*\n";

inline constexpr const char* K_DISABLE_VARIABLE = "HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS";

// Searches a NUL-separated environment block for an assignment to `name`.
//
// `buffer` is the raw /proc/self/environ layout: "A=1\0B=2\0". It needs no terminator of its own --
// every read is bounded by `length` -- so a truncated block is scanned safely rather than running
// off the end.
inline bool environBufferHasFlag(const char* buffer, long length, const char* name)
{
    if(buffer == nullptr || name == nullptr || length <= 0)
    {
        return false;
    }

    for(long i = 0; i < length;)
    {
        long matched = 0;
        while(i + matched < length && name[matched] != '\0' && buffer[i + matched] == name[matched])
        {
            ++matched;
        }
        // Require the '=' so a longer variable sharing this prefix does not match.
        if(name[matched] == '\0' && i + matched < length && buffer[i + matched] == '=')
        {
            return true;
        }
        while(i < length && buffer[i] != '\0')
        {
            ++i;
        }
        ++i;
    }
    return false;
}

#if defined(__linux__)

// An environment larger than this is scanned only up to the limit, so a variable beyond it is not
// seen. Raising it costs stack in a start-up path that has no allocator available.
inline constexpr std::size_t K_ENVIRON_BUFFER_SIZE = 8192;

// Reads the environment as the kernel recorded it at exec.
//
// That snapshot is what /proc/self/environ exposes, so a variable introduced later with setenv()
// does not appear here. The override therefore has to be set before the process starts.
inline bool environmentFlagSet(const char* name)
{
    // Deliberately left uninitialized: value-initializing 8 KB emits a memset call, and memset is
    // one of the interceptors that is not yet wired up when this runs.
    std::array<char, K_ENVIRON_BUFFER_SIZE>
        buffer; // NOLINT(cppcoreguidelines-pro-type-member-init)

    const long fd = syscall(SYS_openat, AT_FDCWD, "/proc/self/environ", O_RDONLY, 0);
    if(fd < 0)
    {
        return false;
    }
    const long length = syscall(SYS_read, fd, buffer.data(), buffer.size());
    syscall(SYS_close, fd);

    return environBufferHasFlag(buffer.data(), length, name);
}

#endif // __linux__

// The suppression text the ASan hook hands back.
//
// The HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS override is Linux-only. Reading the environment this
// early needs a platform-specific route that cannot be exercised without a Windows machine, and
// getting it wrong costs every ASAN binary a silent start-up crash, so elsewhere the suppressions
// are unconditional and disabling them needs a rebuild.
inline const char* defaultSuppressions()
{
#if defined(__linux__)
    if(environmentFlagSet(K_DISABLE_VARIABLE))
    {
        return "";
    }
#endif
    return K_DEFAULT_SUPPRESSIONS;
}

} // namespace hipdnn_test_sdk::utilities::asan

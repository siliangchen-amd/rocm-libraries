// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_test_sdk/utilities/AsanDefaultSuppressions.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>

using namespace hipdnn_test_sdk::utilities;
using hipdnn_test_sdk::utilities::asan::environBufferHasFlag;

namespace
{

// Builds a /proc/self/environ-style block: entries joined by NUL, each entry NUL-terminated.
std::string environBlock(std::initializer_list<std::string_view> entries)
{
    std::string block;
    for(const auto& entry : entries)
    {
        block.append(entry);
        block.push_back('\0');
    }
    return block;
}

bool hasFlag(const std::string& block, const char* name)
{
    return environBufferHasFlag(block.data(), static_cast<long>(block.size()), name);
}

} // namespace

TEST(TestAsanDefaultSuppressions, RejectsNullBuffer)
{
    EXPECT_FALSE(environBufferHasFlag(nullptr, 10, "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsNullName)
{
    const std::string block = environBlock({"FOO=1"});
    EXPECT_FALSE(environBufferHasFlag(block.data(), static_cast<long>(block.size()), nullptr));
}

TEST(TestAsanDefaultSuppressions, RejectsEmptyBuffer)
{
    const std::string block = environBlock({"FOO=1"});
    EXPECT_FALSE(environBufferHasFlag(block.data(), 0, "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsNegativeLength)
{
    // syscall() reports a failed read as -1; that must not be treated as a length.
    const std::string block = environBlock({"FOO=1"});
    EXPECT_FALSE(environBufferHasFlag(block.data(), -1, "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesSoleEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesFirstEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1", "BAR=2", "BAZ=3"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesMiddleEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1", "BAR=2", "BAZ=3"}), "BAR"));
}

TEST(TestAsanDefaultSuppressions, MatchesLastEntry)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=1", "BAR=2", "BAZ=3"}), "BAZ"));
}

TEST(TestAsanDefaultSuppressions, MatchesEmptyValue)
{
    // The override is set for its presence, not its value, so "VAR=" must count as set.
    EXPECT_TRUE(hasFlag(environBlock({"FOO="}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesValueContainingEquals)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO=a=b=c"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsAbsentName)
{
    EXPECT_FALSE(hasFlag(environBlock({"FOO=1", "BAR=2"}), "BAZ"));
}

TEST(TestAsanDefaultSuppressions, RejectsEntryWhoseNameExtendsTheSearchName)
{
    // The '=' requirement is what stops HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS_EXTRA from enabling the
    // override.
    EXPECT_FALSE(hasFlag(environBlock({"FOO_BAR=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsEntryWhoseNameIsAPrefixOfTheSearchName)
{
    EXPECT_FALSE(hasFlag(environBlock({"FOO=1"}), "FOO_BAR"));
}

TEST(TestAsanDefaultSuppressions, RejectsNameAppearingOnlyInsideAValue)
{
    EXPECT_FALSE(hasFlag(environBlock({"BAR=FOO=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, RejectsEntryWithNoAssignment)
{
    EXPECT_FALSE(hasFlag(environBlock({"FOO"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, MatchesAfterAnEntryWithNoAssignment)
{
    EXPECT_TRUE(hasFlag(environBlock({"FOO", "FOO=1"}), "FOO"));
}

TEST(TestAsanDefaultSuppressions, ScansTruncatedBlockWithoutOverrunning)
{
    // A block cut mid-entry has no trailing NUL. Every read is bounded by the length, so the
    // partial tail is simply not a match.
    const std::string block = environBlock({"FOO=1", "BARBAR=2"});
    const auto truncated = static_cast<long>(block.size() - 4);

    EXPECT_TRUE(environBufferHasFlag(block.data(), truncated, "FOO"));
    EXPECT_FALSE(environBufferHasFlag(block.data(), truncated, "BARBAR"));
}

TEST(TestAsanDefaultSuppressions, RejectsTrailingNameWithoutItsEquals)
{
    const char* block = "FOO";
    EXPECT_FALSE(environBufferHasFlag(block, 3, "FOO"));
}

#if defined(__linux__)

TEST(TestAsanDefaultSuppressions, ReadsTheRealEnvironment)
{
    if(std::getenv("PATH") == nullptr)
    {
        GTEST_SKIP() << "PATH is not set in this environment";
    }
    EXPECT_TRUE(asan::environmentFlagSet("PATH"));
}

TEST(TestAsanDefaultSuppressions, RealEnvironmentDoesNotReportAnAbsentVariable)
{
    EXPECT_FALSE(asan::environmentFlagSet("HIPDNN_ASAN_DEFINITELY_NOT_SET_XYZZY"));
}

TEST(TestAsanDefaultSuppressions, SetenvDoesNotChangeTheSnapshot)
{
    // /proc/self/environ is what the kernel recorded at exec, so setenv() cannot enable the
    // override from inside a running process. Anyone tempted to test the override that way needs a
    // subprocess instead.
    const ScopedEnvironmentVariableSetter setter("HIPDNN_ASAN_SET_AFTER_EXEC_XYZZY", "1");

    ASSERT_NE(std::getenv("HIPDNN_ASAN_SET_AFTER_EXEC_XYZZY"), nullptr);
    EXPECT_FALSE(asan::environmentFlagSet("HIPDNN_ASAN_SET_AFTER_EXEC_XYZZY"));
}

TEST(TestAsanDefaultSuppressions, ReturnsEmptyWhenTheOverrideIsSetBeforeExec)
{
    // Only reachable when the whole process was started with the variable set, which is the
    // documented way to use it. The suite is run a second time that way so this case is covered.
    if(!asan::environmentFlagSet(asan::K_DISABLE_VARIABLE))
    {
        GTEST_SKIP() << "set " << asan::K_DISABLE_VARIABLE << " before exec to exercise this path";
    }
    EXPECT_STREQ(asan::defaultSuppressions(), "");
}

#endif // __linux__

TEST(TestAsanDefaultSuppressions, SuppressionTextNamesTheTensileEntryPoint)
{
    EXPECT_STREQ(asan::K_DEFAULT_SUPPRESSIONS, "interceptor_via_fun:*findBestKeyMatch*\n");
}

TEST(TestAsanDefaultSuppressions, ReturnsTheSuppressionTextByDefault)
{
#if defined(__linux__)
    if(asan::environmentFlagSet(asan::K_DISABLE_VARIABLE))
    {
        GTEST_SKIP() << asan::K_DISABLE_VARIABLE << " is set, so the suppressions are disabled";
    }
#endif
    EXPECT_STREQ(asan::defaultSuppressions(), asan::K_DEFAULT_SUPPRESSIONS);
}

#ifdef ADDRESS_SANITIZER

// Declared rather than included: the ASan runtime owns this symbol and ships no header for it.
// The reserved name is the runtime's, not ours -- it only resolves under exactly this spelling.
// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __asan_default_suppressions();

TEST(TestAsanDefaultSuppressions, AsanRuntimeHookResolvesToOurDefinition)
{
    // The ASan runtime provides a weak default returning "". Seeing our text here proves the
    // strong definition in this executable is the one the runtime found -- the property that the
    // whole mechanism depends on and that no other test covers.
    EXPECT_STREQ(__asan_default_suppressions(), asan::defaultSuppressions());

#if defined(__linux__)
    if(asan::environmentFlagSet(asan::K_DISABLE_VARIABLE))
    {
        GTEST_SKIP() << asan::K_DISABLE_VARIABLE << " is set, so the suppressions are disabled";
    }
#endif
    EXPECT_STREQ(__asan_default_suppressions(), asan::K_DEFAULT_SUPPRESSIONS);
}

#endif // ADDRESS_SANITIZER

"""
Makes a system-installed Expat XML parser headers and library available to be
used as a C/C++ dependency. On Ubuntu, pkg-config is used to locate the Expat
headers and library. On macOS, no pkg-config expat.pc file is installed, but
the Expat headers are included in the macOS SDK and the library is always
located at /usr/lib.

Example:
    WORKSPACE:
        load("@drake//tools/workspace/expat:repository.bzl", "expat_repository")  # noqa
        expat_repository(name = "foo")

    BUILD:
        cc_library(
            name = "foobar",
            deps = ["@foo//:expat"],
            srcs = ["bar.cc"],
        )

Argument:
    name: A unique name for this rule.
"""

load("//tools/workspace:execute.bzl", "execute_or_fail")
load("//tools/skylark:pathutils.bzl", "join_paths")
load("//tools/workspace:pkg_config.bzl", "setup_pkg_config_repository")
load("//tools/workspace:os.bzl", "determine_os")

def _impl(repository_ctx):
    os_result = determine_os(repository_ctx)
    if os_result.error != None:
        fail(os_result.error)

    if os_result.is_macos:
        result = execute_or_fail(repository_ctx, ["xcrun", "--show-sdk-path"])
        include = join_paths(result.stdout.strip(), "usr/include")
        repository_ctx.symlink(
            join_paths(include, "expat.h"),
            "include/expat.h",
        )
        repository_ctx.symlink(
            join_paths(include, "expat_external.h"),
            "include/expat_external.h",
        )

        file_content = """# DO NOT EDIT: generated by expat_repository()

load("@drake//tools/skylark:cc.bzl", "cc_library")

licenses(["notice"])  # MIT

cc_library(
    name = "expat",
    hdrs = [
        "include/expat_external.h",
        "include/expat.h",
    ],
    includes = ["include"],
    linkopts = ["-lexpat"],
    visibility = ["//visibility:public"],
    deprecation = "The @expat external is deprecated in Drake's WORKSPACE and will be removed on or after 2024-02-01.",  # noqa
)
"""

        repository_ctx.file(
            "BUILD.bazel",
            content = file_content,
            executable = False,
        )
    else:
        error = setup_pkg_config_repository(repository_ctx).error

        if error != None:
            fail(error)

expat_repository = repository_rule(
    # TODO(jamiesnape): Pass down licenses to setup_pkg_config_repository.
    attrs = {
        "modname": attr.string(default = "expat"),
        "extra_deprecation": attr.string(default = "The @expat external is deprecated in Drake's WORKSPACE and will be removed on or after 2024-02-01."),  # noqa
    },
    local = True,
    configure = True,
    implementation = _impl,
)

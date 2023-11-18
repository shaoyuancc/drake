#!/usr/bin/env python3
"""
upgrade.py - Upgrades the lockfile for Drake's Rust toolchain.

This program is only tested / supported on Ubuntu.

This downloads a lot of stuff. Be sure you're on a good network and have
`--repository_cache=...` enabled in your `$HOME/.bazelrc`.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Operate relative to this script in the source tree.
    os.chdir(Path(__file__).resolve().parent)

    logging.info("Downloading ALL Rust toolchains via `bazel sync`.")
    output_base = Path("upgrade.output_base")
    if output_base.exists():
        shutil.rmtree(output_base)
    subprocess.run(cwd="upgrade", check=True, args=[
        "bazel", "--output_base=../upgrade.output_base", "sync"])
    external = output_base / "external"

    # Identify the repository names we want.
    repository_names = []
    for item in external.iterdir():
        if item.name.startswith("rust_") and "__stable" in item.name:
            repository_names.append(item.name)
    assert repository_names
    repository_names = sorted(repository_names)

    # Add a breadcrumb to our generated files.
    prologue = "# This file is automatically generated by upgrade.py.\n\n"

    # Copy the generated BUILD files into our lock/details directory, while
    # also removing any BUILD files that are no longer relevant.
    details = Path("lock/details")
    shutil.rmtree(details)
    details.mkdir()
    for name in repository_names:
        content = (external / name / "BUILD.bazel").read_text(encoding="utf-8")
        dest = details / f"BUILD.{name}.bazel"
        logging.info(f"Copying {dest}")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(prologue)
            f.write(content)

    # Concatenate the instrumented download_and_extract metadata.
    archives = Path("lock/archives.bzl")
    logging.info(f"Writing {archives}")
    with open(archives, "w", encoding="utf-8") as f:
        f.write(prologue)
        f.write("ARCHIVES = [\n")
        for name in repository_names:
            json_path = external / name / "download_and_extract.json"
            f.write("dict(\n")
            f.write(f"name = \"{name}\",\n")
            label = (
                "@drake//tools/workspace/rust_toolchain:"
                + f"lock/details/BUILD.{name}.bazel"
            )
            f.write(f"build_file = Label(\"{label}\"),\n")
            if json_path.exists():
                json_text = json_path.read_text(encoding="utf-8")
                json_text_pretty = json.dumps(json.loads(json_text), indent=2)
                f.write("downloads = json.encode(\n")
                f.write(json_text_pretty)
                f.write("),\n")
                pass
            else:
                f.write("downloads = \"[]\",\n")
            f.write("),\n")
        f.write("]\n")
    subprocess.run(check=True, args=[
        "bazel", "build", "//tools/lint:buildifier"])
    subprocess.run(check=True, args=[
        "../../../bazel-bin/tools/lint/buildifier", archives])

    # Clean up (but not with a context manager -- if we crash, we should leave
    # the intermediate output intact for debugging).
    shutil.rmtree(output_base)


assert __name__ == "__main__"
_main()

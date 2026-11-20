#!/usr/bin/env python3
"""Staged releases and validated Docker artifacts; no third-party Python dependencies."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "docker.jolibrain.com/joligen_server"
BUILD_IMAGE = "docker.jolibrain.com/joligen_build"
COMMAND_LOG = None


def run(*args, cwd=None, capture=False, timeout=None):
    command = shlex.join(map(str, args))
    print("+ " + command, flush=True)
    if COMMAND_LOG is not None and not capture:
        with COMMAND_LOG.open("a") as log:
            log.write("\n+ " + command + "\n")
            log.flush()
            try:
                subprocess.run(
                    list(map(str, args)),
                    cwd=cwd,
                    check=True,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.SubprocessError:
                print(
                    "\n".join(
                        COMMAND_LOG.read_text(errors="replace").splitlines()[-30:]
                    ),
                    file=sys.stderr,
                )
                raise
        return None
    return subprocess.run(
        list(map(str, args)),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        timeout=timeout,
    ).stdout


def git(*args, cwd=ROOT):
    return run("git", *args, cwd=cwd, capture=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def notes(changelog, version):
    match = re.search(
        r"^#{2,3} \[" + re.escape(version) + r"\][^\n]*\n(.*?)(?=^#{2,3} \[|\Z)",
        changelog,
        re.M | re.S,
    )
    if not match or not match[1].strip():
        raise RuntimeError("Generated release notes are empty")
    return match[1].strip()


def version_key(value):
    core, separator, prerelease = value.partition("-")
    parts = (
        tuple((0, int(p)) if p.isdigit() else (1, p) for p in prerelease.split("."))
        if separator
        else ()
    )
    return (*map(int, core.split(".")), not separator, parts)


def gpu_args():
    return shlex.split(os.environ.get("DOCKER_GPU_RUN_ARGS", "--gpus all"))


def image_id(image):
    return run(
        "docker", "image", "inspect", "--format", "{{.Id}}", image, capture=True
    ).strip()


def build(tag, report_dir):
    global COMMAND_LOG
    report_dir.mkdir(parents=True, exist_ok=True)
    COMMAND_LOG = report_dir / "build.log"
    commit = git("rev-parse", "HEAD")
    image = f"{IMAGE}:{tag}"
    report = {"tag": tag, "source_commit": commit, "image": image, "status": "building"}
    report_file = report_dir / "artifact.json"
    save(report_file, report)
    container = "joligen-release-" + uuid.uuid4().hex[:12]
    try:
        # Archive only committed inputs. In particular, never send the caller's
        # credentials, untracked experiments, or installed node_modules to Docker.
        with tempfile.TemporaryDirectory(prefix="joligen-context-") as tmp:
            context = Path(tmp)
            archive = context / "source.tar"
            run("git", "archive", "--format=tar", "-o", archive, "HEAD", cwd=ROOT)
            run("tar", "-xf", archive, "-C", context)
            archive.unlink()
            run(
                "docker",
                "build",
                "-t",
                f"{BUILD_IMAGE}:{tag}",
                "-f",
                "docker/Dockerfile.build",
                "--build-arg",
                f"JOLIGEN_VERSION={tag.removeprefix('v')}",
                "--build-arg",
                f"JOLIGEN_COMMIT={commit}",
                ".",
                cwd=context,
            )
            run(
                "docker",
                "build",
                "-t",
                image,
                "-f",
                "docker/Dockerfile.server",
                "--build-arg",
                f"GTAG={tag}",
                ".",
                cwd=context,
            )
        report["image_id"] = image_id(image)
        # Use immutable IDs for all tests and subsequent publication checks.
        tested = report["image_id"]
        report["dependencies"] = run(
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            tested,
            "-m",
            "pip",
            "freeze",
            capture=True,
        )
        report["status"] = "smoke-testing"
        save(report_file, report)
        run(
            "docker",
            "run",
            "--rm",
            "--name",
            container + "-cuda",
            *gpu_args(),
            "--entrypoint",
            "python3",
            tested,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "assert (torch.ones(4, device='cuda') * 2).sum().item() == 8; "
            "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())",
            timeout=120,
        )
        run(
            "docker",
            "run",
            "--rm",
            "--name",
            container + "-cli",
            "--entrypoint",
            "python3",
            tested,
            "train.py",
            "--help",
            timeout=180,
        )
        run("docker", "run", "-d", "--name", container, *gpu_args(), tested)
        deadline = time.monotonic() + 180
        probe = (
            "import json, urllib.request; "
            "info=json.load(urllib.request.urlopen('http://127.0.0.1:8000/info', timeout=3)); "
            f"assert {commit!r} in info['JoliGEN']; "
            "schema=json.load(urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=3)); "
            f"assert schema['info']['version'] == {tag.removeprefix('v')!r}"
        )
        while True:
            try:
                run("docker", "exec", container, "python3", "-c", probe, timeout=10)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Server startup timed out")
                time.sleep(2)
        report["smoke_tests"] = "passed"
        report["status"] = "suite-running"
        save(report_file, report)
        # No host bind mounts: the suite's destructive cleanup only sees a fresh
        # directory inside this disposable container.
        run(
            "docker",
            "run",
            "--name",
            container + "-suite",
            *gpu_args(),
            "--shm-size=8gb",
            "--entrypoint",
            "bash",
            tested,
            "-c",
            "set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1; "
            "d=$(mktemp -d /tmp/joligen-tests.XXXXXX); "
            'bash scripts/run_tests.sh "$d"',
            timeout=21600,
        )
        report["full_suite"] = "passed"
        report["status"] = "validated"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        save(report_file, report)
        for suffix in ("-cuda", "-cli"):
            subprocess.run(
                ["docker", "rm", "-f", container + suffix],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for name, log_name in [
            (container, "server.log"),
            (container + "-suite", "suite.log"),
        ]:
            with (report_dir / log_name).open("w") as log:
                subprocess.run(
                    ["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT
                )
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return report_file


def push(tag, report_file):
    report = json.loads(report_file.read_text())
    image = f"{IMAGE}:{tag}"
    if (
        report.get("status") != "validated"
        or report.get("tag") != tag
        or report.get("image") != image
        or report.get("image_id") != image_id(image)
    ):
        raise RuntimeError(
            "Refusing to push: image does not match a fully validated artifact report"
        )
    # Push by a checked local tag; latest is updated only after version upload.
    run("docker", "push", image)
    if "-" not in tag:
        run("docker", "tag", report["image_id"], f"{IMAGE}:latest")
        run("docker", "push", f"{IMAGE}:latest")


def remote_ready(parent, tag, cwd):
    run("git", "fetch", "origin", "master", "--tags", cwd=cwd)
    if git("rev-parse", "origin/master", cwd=cwd) != parent:
        raise RuntimeError("origin/master moved; restart staging before publication")
    if git("ls-remote", "--tags", "origin", f"refs/tags/{tag}", cwd=cwd):
        raise RuntimeError(f"Remote tag {tag} already exists")


def release(args):
    global COMMAND_LOG
    selection = args.version
    if not re.fullmatch(
        r"(?:major|minor|patch|premajor|preminor|prepatch|prerelease|\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)",
        selection,
    ):
        raise RuntimeError("Expected an explicit version (without v) or a version bump")
    branch = git("branch", "--show-current")
    if not args.dry_run and (branch != "master" or git("status", "--porcelain")):
        raise RuntimeError("Publication requires a clean master checkout")
    if args.dry_run and git("status", "--porcelain"):
        print(
            "Using committed HEAD only; working-tree changes are excluded.", flush=True
        )
    origin = git("remote", "get-url", "origin")
    parent = git("rev-parse", "HEAD")
    stage = Path(
        tempfile.mkdtemp(prefix="joligen-release-", dir=args.stage_root)
    ).resolve()
    repo = stage / "repo"
    COMMAND_LOG = stage / "commands.log"
    report = {"source_commit": parent, "dry_run": args.dry_run, "status": "preparing"}
    report_path = stage / "release-report.json"
    print(f"Release staging and reports: {stage}", flush=True)
    succeeded = False
    try:
        run("git", "clone", "--local", "--no-hardlinks", ROOT, repo)
        run("git", "checkout", "--detach", parent, cwd=repo)
        run("git", "remote", "set-url", "origin", origin, cwd=repo)
        # Clones do not inherit repository-local author configuration.
        for key in ("user.name", "user.email"):
            value = git("config", key)
            run("git", "config", key, value, cwd=repo)
        if not args.dry_run:
            run("gh", "auth", "status", cwd=repo)
            run("git", "fetch", "origin", "master", "--tags", cwd=repo)
            if git("rev-parse", "origin/master", cwd=repo) != parent:
                raise RuntimeError("Local master is not equal to origin/master")
        # npm is used only for the existing standard-version development tool.
        run(
            "npm",
            "install",
            "--ignore-scripts",
            "--no-package-lock",
            "--no-audit",
            "--no-fund",
            cwd=repo,
        )
        old_version = json.loads((repo / "package.json").read_text())["version"]
        run(
            repo / "node_modules/.bin/standard-version",
            "--release-as",
            selection,
            cwd=repo,
        )
        version = json.loads((repo / "package.json").read_text())["version"]
        if version_key(version) <= version_key(old_version):
            raise RuntimeError("Release version did not advance")
        tag = "v" + version
        commit = git("rev-parse", "HEAD", cwd=repo)
        if git("rev-list", "-n", "1", tag, cwd=repo) != commit:
            raise RuntimeError("Release tag does not point to the release commit")
        if not args.dry_run:
            remote_ready(parent, tag, repo)
        highlight = repo / "ci/release-highlights" / f"{version}.md"
        body = (highlight.read_text().strip() + "\n\n") if highlight.exists() else ""
        body += "### Changes\n\n" + notes((repo / "CHANGELOG.md").read_text(), version)
        body += f"\n\n### Docker image\n\n```bash\ndocker pull {IMAGE}:{tag}\n"
        body += f"docker run --rm --gpus all -p 8000:8000 {IMAGE}:{tag}\n```\n"
        note_path = stage / "release-notes.md"
        note_path.write_text(body)
        report.update(
            version=version,
            tag=tag,
            release_commit=commit,
            notes=str(note_path),
            status="validating",
        )
        save(report_path, report)
        run(
            "bash",
            repo / "ci/build-docker-images.sh",
            "--tag",
            tag,
            "--no-push",
            "--report-dir",
            stage / "validation",
            cwd=repo,
        )
        report["artifact_report"] = str(stage / "validation/artifact.json")
        report["status"] = "validated"
        save(report_path, report)
        print(body, flush=True)
        if args.dry_run:
            print(
                "Dry run complete: no images, Git refs, or GitHub releases published."
            )
        else:
            if not args.yes and input(
                f"Publish {tag} and create a draft GitHub release? [y/N] "
            ).lower() not in ("y", "yes"):
                raise RuntimeError("Publication cancelled; validated stage retained")
            remote_ready(parent, tag, repo)
            run(
                "bash",
                repo / "ci/build-docker-images.sh",
                "--tag",
                tag,
                "--push-only",
                "--report-dir",
                stage / "validation",
                cwd=repo,
            )
            report["status"] = "images-published"
            save(report_path, report)
            run(
                "git",
                "push",
                "--atomic",
                "origin",
                "HEAD:refs/heads/master",
                f"refs/tags/{tag}",
                cwd=repo,
            )
            report["status"] = "refs-published"
            save(report_path, report)
            run(
                "gh",
                "release",
                "create",
                tag,
                "--title",
                f"joliGEN {tag}",
                "--notes-file",
                note_path,
                "--draft",
                "--verify-tag",
                cwd=repo,
            )
            report["status"] = "draft-created"
        succeeded = True
    except BaseException as exc:
        report["error"] = str(exc)
        report["failed_at"] = report["status"]
        report["status"] = "failed"
        raise
    finally:
        save(report_path, report)
        # Always retain reports. Dry runs retain their candidate for inspection.
        if succeeded and not args.keep_stage and not args.dry_run:
            shutil.rmtree(repo)
        print(f"Release report retained at {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rel = commands.add_parser(
        "release", help="Stage, validate, then optionally publish"
    )
    rel.add_argument(
        "version", help="Explicit X.Y.Z or major/minor/patch/prerelease bump"
    )
    rel.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and test without publishing anything",
    )
    rel.add_argument(
        "--yes", action="store_true", help="Skip the final publication prompt"
    )
    rel.add_argument("--keep-stage", action="store_true")
    rel.add_argument("--stage-root", default=tempfile.gettempdir())
    dock = commands.add_parser(
        "docker", help="Build/test a committed checkout or push a validated image"
    )
    dock.add_argument("--tag", required=True)
    mode = dock.add_mutually_exclusive_group(required=True)
    mode.add_argument("--no-push", action="store_true")
    mode.add_argument("--push-only", action="store_true")
    dock.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "release":
        release(args)
    else:
        if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", args.tag):
            parser.error("--tag must be a version tag such as v5.0.0")
        if args.push_only:
            push(args.tag, args.report_dir / "artifact.json")
        else:
            build(args.tag, args.report_dir)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        print(f"Release failed: {exc}", file=sys.stderr)
        sys.exit(1)

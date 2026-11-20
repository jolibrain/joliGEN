"""Release integration tests using real temporary Git repos and fake services.

Run without the ML environment: python3 -m unittest discover -s tests -p test_release_workflow.py
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

VERSIONER = """#!/usr/bin/env python3
import json, pathlib, subprocess, sys
p = pathlib.Path('package.json')
data = json.loads(p.read_text())
version = sys.argv[-1]
if version == 'major': version = '5.0.0'
data['version'] = version
p.write_text(json.dumps(data))
c = pathlib.Path('CHANGELOG.md')
c.write_text('# Changelog\\n\\n## [' + version + '] (today)\\n\\n* New feature\\n\\n' + c.read_text())
subprocess.run(['git', 'add', 'package.json', 'CHANGELOG.md'], check=True)
subprocess.run(['git', 'commit', '-m', 'release'], check=True)
subprocess.run(['git', 'tag', 'v' + version], check=True)
"""
FAKE = """#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['FAKE_LOG'], 'a') as f: f.write(json.dumps([name] + args) + '\\n')
if name == 'npm':
    p = pathlib.Path('node_modules/.bin/standard-version')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(os.environ['VERSIONER'])
    p.chmod(0o755)
if name == 'docker':
    failure = os.environ.get('FAIL_AT')
    if failure == 'build' and args[0] == 'build': sys.exit(1)
    if failure == 'cuda' and 'assert torch.cuda.is_available()' in ' '.join(args): sys.exit(1)
    if failure == 'suite' and 'bash scripts/run_tests.sh' in ' '.join(args): sys.exit(1)
    if args[:2] == ['image', 'inspect']: print(os.environ.get('IMAGE_ID', 'sha256:test'))
    if 'freeze' in args: print('torch==2.4.0')
    if args[0] == 'build' and os.environ.get('MOVE_REMOTE'):
        remote = os.environ['MOVE_REMOTE']
        parent = subprocess.check_output(['git', '--git-dir', remote, 'rev-parse', 'master'], text=True).strip()
        tree = subprocess.check_output(['git', '--git-dir', remote, 'rev-parse', 'master^{tree}'], text=True).strip()
        env = dict(os.environ, GIT_AUTHOR_NAME='Test', GIT_AUTHOR_EMAIL='test@example.org', GIT_COMMITTER_NAME='Test', GIT_COMMITTER_EMAIL='test@example.org')
        commit = subprocess.check_output(['git', '--git-dir', remote, 'commit-tree', tree, '-p', parent, '-m', 'concurrent change'], text=True, env=env).strip()
        subprocess.run(['git', '--git-dir', remote, 'update-ref', 'refs/heads/master', commit], check=True)
"""


class ReleaseWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.remote = self.base / "remote.git"
        self.stages = self.base / "stages"
        self.stages.mkdir()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.log = self.base / "calls.jsonl"
        self.env = dict(
            os.environ,
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
            FAKE_LOG=str(self.log),
            VERSIONER=VERSIONER,
        )
        for name in ("npm", "docker", "gh"):
            p = self.bin / name
            p.write_text(FAKE)
            p.chmod(0o755)
        self.git("init", "-b", "master")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "test@example.org")
        (self.repo / "ci").mkdir()
        for name in ("release.py", "build-docker-images.sh"):
            shutil.copy2(ROOT / "ci" / name, self.repo / "ci" / name)
        (self.repo / "package.json").write_text('{"version":"4.0.0"}')
        (self.repo / "CHANGELOG.md").write_text("## [4.0.0]\n\n* Historical change\n")
        self.git("add", ".")
        self.git("commit", "-m", "initial")
        self.git("tag", "v4.0.0")
        subprocess.run(
            ["git", "init", "--bare", str(self.remote)], check=True, capture_output=True
        )
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "origin", "master", "--tags")
        self.head = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL
        ).strip()

    def release(self, *args, **env):
        result = subprocess.run(
            [
                "python3",
                "ci/release.py",
                "release",
                *args,
                "--stage-root",
                str(self.stages),
            ],
            cwd=self.repo,
            env=dict(self.env, **env),
            text=True,
            capture_output=True,
        )
        return result

    def calls(self):
        return [json.loads(s) for s in self.log.read_text().splitlines()]

    def report(self):
        return json.loads(next(self.stages.glob("*/release-report.json")).read_text())

    def assert_unpublished(self):
        self.assertFalse(any(c[0] == "docker" and c[1] == "push" for c in self.calls()))
        self.assertFalse(
            any(c[:3] == ["gh", "release", "create"] for c in self.calls())
        )
        self.assertEqual(
            self.git("ls-remote", "origin", "refs/heads/master").split()[0], self.head
        )
        self.assertEqual(self.git("rev-parse", "HEAD"), self.head)

    def test_dry_run_isolated_and_preserves_history(self):
        (self.repo / "secret.txt").write_text("untracked")
        result = self.release("--dry-run", "5.0.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_unpublished()
        self.assertEqual(self.report()["status"], "validated")
        candidate = next(self.stages.glob("*/repo"))
        self.assertFalse((candidate / "secret.txt").exists())
        self.assertIn("Historical change", (candidate / "CHANGELOG.md").read_text())
        self.assertEqual(self.git("tag", "--list", "v5.0.0"), "")

    def test_build_smoke_and_suite_failures_block_publication(self):
        for step in ("build", "cuda", "suite"):
            with self.subTest(step=step):
                result = self.release("--yes", "5.0.0", FAIL_AT=step)
                self.assertNotEqual(result.returncode, 0)
                self.assert_unpublished()

    def test_publish_after_validation(self):
        result = self.release("--yes", "major")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        suite = next(
            i for i, c in enumerate(calls) if "bash scripts/run_tests.sh" in " ".join(c)
        )
        push = next(i for i, c in enumerate(calls) if c[:2] == ["docker", "push"])
        draft = next(
            i for i, c in enumerate(calls) if c[:3] == ["gh", "release", "create"]
        )
        self.assertLess(suite, push)
        self.assertLess(push, draft)
        self.assertEqual(self.report()["status"], "draft-created")
        self.assertTrue(self.git("ls-remote", "origin", "refs/tags/v5.0.0"))
        self.assertEqual(self.git("rev-parse", "HEAD"), self.head)

    def test_remote_movement_blocks_publication(self):
        result = self.release("--yes", "5.0.0", MOVE_REMOTE=str(self.remote))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("origin/master moved", result.stderr)
        self.assertFalse(any(c[:2] == ["docker", "push"] for c in self.calls()))

    def test_tag_collision_blocks_publication(self):
        self.git("tag", "v5.0.0")
        self.git("push", "origin", "v5.0.0")
        result = self.release("--yes", "5.0.0")
        self.assertNotEqual(result.returncode, 0)
        self.assert_unpublished()

    def test_changed_image_cannot_be_pushed(self):
        result = self.release("--dry-run", "5.0.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        report_dir = next(self.stages.glob("*/validation"))
        result = subprocess.run(
            [
                "python3",
                "ci/release.py",
                "docker",
                "--tag",
                "v5.0.0",
                "--push-only",
                "--report-dir",
                str(report_dir),
            ],
            cwd=self.repo,
            env=dict(self.env, IMAGE_ID="sha256:replacement"),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assert_unpublished()

    def test_atomic_ref_failure_keeps_branch_and_stage(self):
        hook = self.remote / "hooks/pre-receive"
        hook.write_text(
            '#!/bin/sh\nwhile read old new ref; do\ncase "$ref" in refs/tags/v5.0.0) exit 1;; esac\ndone\n'
        )
        hook.chmod(0o755)
        result = self.release("--yes", "5.0.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            self.git("ls-remote", "origin", "refs/heads/master").split()[0], self.head
        )
        self.assertFalse(self.git("ls-remote", "origin", "refs/tags/v5.0.0"))
        self.assertEqual(self.report()["failed_at"], "images-published")
        self.assertTrue(next(self.stages.glob("*/repo")).exists())
        self.assertFalse(
            any(c[:3] == ["gh", "release", "create"] for c in self.calls())
        )

    def test_downgrade_is_rejected(self):
        result = self.release("--dry-run", "3.0.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not advance", result.stderr)
        self.assert_unpublished()

    def test_dirty_publication_is_rejected(self):
        (self.repo / "untracked").touch()
        result = self.release("--yes", "5.0.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("clean master", result.stderr)


if __name__ == "__main__":
    unittest.main()

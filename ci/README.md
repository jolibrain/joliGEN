# Release process

The release workflow builds from committed Git inputs in an isolated clone. It
uses Python 3.9+, Git, Node/npm, Docker with the NVIDIA container runtime, and
(for publication) authenticated `gh` and registry access. Git author identity
must be configured. GPU validation downloads the datasets and model weights
used by `scripts/run_tests.sh`; allow several hours and ample disk space.

## Rehearse v5.0.0

Commit the intended release changes on your working branch, then run:

```bash
DOCKER_GPU_RUN_ARGS='--gpus device=1' ci/release_all.sh --dry-run 5.0.0
```

A dry run uses committed HEAD only, even if the checkout is dirty. It creates
a local release commit and tag **inside the staging clone**, builds the server
image, tests CUDA execution, CLI startup, `/info`, and `/openapi.json`, and runs
the full existing Jenkins suite inside a disposable container. No host dataset
or checkpoint directories are mounted. The test suite's cleanup is confined to
that container. There is no option to skip validation in a release.

No Docker images, remote Git refs, or GitHub releases are published in a dry
run. The default GPU selection is `--gpus all`; set `DOCKER_GPU_RUN_ARGS` to
select an available GPU. The workflow does not reserve GPUs across users.

The printed staging directory contains `repo/`, `release-notes.md`,
`release-report.json`, `commands.log`, and `validation/artifact.json`,
`build.log`, `server.log`, and `suite.log`. Artifact metadata records the tested image ID and installed Python
dependencies. The candidate and reports remain available after dry runs or
failures. Use `--stage-root DIR` to select a disk with enough space.

The v5 highlights live in `ci/release-highlights/5.0.0.md`. Future versions may
add a corresponding file. Generated conventional-commit changes and Docker
instructions follow these highlights. Existing changelog history is retained.

## Publish later

Merge the reviewed changes into `master`, fetch current refs, and ensure your
checkout is clean and exactly matches `origin/master`. No reset is necessary:

```bash
git fetch origin master --tags
git checkout master
git merge --ff-only origin/master
ci/release_all.sh 5.0.0
```

Alternatively, select a standard-version bump such as `major`, `minor`, or
`patch`. A version selection is mandatory. The release commit updates
`package.json` and `CHANGELOG.md`; the working checkout remains unchanged.

After validation, the script previews the notes and asks for confirmation.
`--yes` suppresses that prompt for trusted automation. Publication rechecks
`origin/master` and tag availability, then:

1. Pushes the validated server image under `vX.Y.Z`, then `latest` (stable
   releases only). The build image is local and is not published.
2. Pushes the release commit and tag in one atomic Git push.
3. Creates a **draft** GitHub release with the prepared notes.

Review and publish the GitHub draft separately. Successful publication retains
reports but removes the staging clone unless `--keep-stage` is supplied. Fetch
and fast-forward your working checkout afterward.

`release.sh` and `ci/release.sh` forward to this workflow. They no longer default
to a major bump or create a source-only release.

## Independent Docker validation

From a committed checkout:

```bash
ci/build-docker-images.sh --tag v5.0.0 --no-push --report-dir /tmp/joligen-validation
ci/build-docker-images.sh --tag v5.0.0 --push-only --report-dir /tmp/joligen-validation
```

The first command runs the same smoke tests and full suite. The second requires
a successful report and an unchanged local image ID; it does not rebuild.
`docker/release_docker.sh` forwards to this helper and requires the same flags.
Build contexts are exported from Git, with additional `.dockerignore` filtering.

## Failures and partial publication

A failed build or test blocks publication. Inspect the retained staging
location and reports, fix the cause, commit the fix, and start a new rehearsal.
Do not reuse a successful report after modifying the image.

Docker, Git, and GitHub do not offer a shared transaction. If publication fails,
inspect `release-report.json`, the registry tags, `git ls-remote origin`, and
`gh release view vX.Y.Z` before recovery. Never overwrite an existing release
Git tag. If only Docker publication completed, retain the exact validated
candidate and check that master still equals its recorded source commit before
pushing its release commit and tag atomically. If Git refs already match the
recorded release commit, create only the missing draft from the retained notes:

```bash
# Run in the retained staging repo after checking remote refs.
gh release create v5.0.0 --title 'joliGEN v5.0.0' \
  --notes-file ../release-notes.md --draft --verify-tag
```

If remote master has diverged, stop recovery and prepare a new reviewed
candidate/version rather than forcing refs. Keep the reports until the release
is complete, then remove staging directories and unused local images manually.

## Tooling tests

```bash
python3 -m unittest discover -s tests -p test_release_workflow.py
```

These tests use real temporary Git repositories and stub npm, Docker, and GitHub
operations. They do not require the ML dependencies or a GPU.

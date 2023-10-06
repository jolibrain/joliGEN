# Release process

## Code release
On a clean `master` branch with all tags fetched:

```bash
$ git fetch --tags
$ git checkout master
$ git reset --hard origin/master
$ yarn
$ ci/release.sh
```

If the result is OK, publish the release note on GitHub and push tags:

```
$ git push --follow-tags origin master
```

The script `ci/release.sh` updates CHANGELOG.md, commits it, creates a tag, and
creates the GitHub release.

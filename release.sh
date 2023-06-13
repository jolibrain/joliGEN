#!/bin/bash

set -e

kind=${1:-major}
git rm CHANGELOG.md
yarn run standard-version -r $kind
tag=$(cat package.json | jq -r .version)

sed -ne "/^## \[$tag\]/,/^##.*202/p" CHANGELOG.md | sed -e '$d' -e '1d' > note.md

cat >> note.md <<EOF
### Docker images:

* joliGEN server: \`docker pull docker.jolibrain.com/joligen_server:v$tag\`
* All images available from https://docker.jolibrain.com/#!/taglist/joligen_server
EOF

trap "rm -f note.md" EXIT
gh release create --title "joliGEN v$tag" -F note.md -d v$tag

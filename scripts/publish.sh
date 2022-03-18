#!/bin/bash
set -e

git config credential.helper "store --file=.git/credentials"
echo "https://${GH_TOKEN}:@github.com" > .git/credentials

git config user.email "andnpatterson@gmail.com"
git config user.name "github-action"

git fetch --unshallow
git fetch --all --tags

git checkout main

# sync package dependencies
pipenv-setup sync --dev --pipfile

# bump the version
cz bump --no-verify --yes --check-consistency

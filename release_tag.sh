#!/bin/bash
set -e

VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: ./release.sh <version>"
  exit 1
fi

echo "Updating dependencies..."
cargo update

echo "Pre-release checks..."
cargo check --all-features
cargo test --all-features
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings

echo "Updating Cargo.toml to $VERSION"
sed -i.bak -E "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
rm Cargo.toml.bak

echo "Post-version build check..."
cargo check --all-features
cargo test --all-features

git add .
git commit -m "Release v$VERSION"

git tag -a "v$VERSION" -m "Release v$VERSION"

git push
git push origin "v$VERSION"

echo "Release v$VERSION complete"

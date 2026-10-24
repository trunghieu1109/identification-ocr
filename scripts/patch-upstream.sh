#!/bin/sh
set -eu

project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
source_dir="${1:?Expected upstream checkout path}"

for patch_file in "$project_dir"/patches/*.patch; do
  if git -C "$source_dir" apply --check "$patch_file" 2>/dev/null; then
    git -C "$source_dir" apply "$patch_file"
  elif ! git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "Cannot apply compatibility patch: $patch_file" >&2
    exit 1
  fi
done

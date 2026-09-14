#!/usr/bin/env bash
# Fetch the TinyStories corpora used by this project.
# Only a byte-range slice of the 1.9 GB training file is taken; the range is
# truncated at the last complete document by the data preparation step.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
B=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
I=https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main

fetch() {  # url, output, [byte-range]
  local url=$1 out=$2 range=${3:-}
  if [ -s "$out" ]; then echo "  have $out"; return; fi
  echo "  fetching $out ${range:+(range $range)}"
  if [ -n "$range" ]; then curl -fsSL -r "$range" "$url" -o "$out"
  else curl -fsSL "$url" -o "$out"; fi
}

echo "TinyStories V2 (pretraining):"
fetch "$B/TinyStoriesV2-GPT4-train.txt" data/raw/stories_train.slice.txt 0-125829119
fetch "$B/TinyStoriesV2-GPT4-valid.txt" data/raw/stories_valid.txt
echo "TinyStories-Instruct (chat SFT):"
fetch "$I/TinyStories-Instruct-train.txt" data/raw/instruct_train.slice.txt 0-62914559
fetch "$I/TinyStories-Instruct-valid.txt" data/raw/instruct_valid.txt
echo "done:"; ls -lh data/raw/*.txt

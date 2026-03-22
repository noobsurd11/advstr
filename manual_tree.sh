#!/bin/bash

# Default directory is current folder
DIR="${1:-.}"

print_tree() {
    local dir="$1"
    local prefix="$2"

    # Loop through items in directory
    for file in "$dir"/*; do
        # Skip if no files
        [ -e "$file" ] || continue

        # Print branch
        echo "${prefix}├── $(basename "$file")"

        # If directory, recurse deeper
        if [ -d "$file" ]; then
            print_tree "$file" "${prefix}│   "
        fi
    done
}

echo "$(basename "$DIR")/"
print_tree "$DIR"

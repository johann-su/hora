#!/bin/bash
# Convert hora's URDF assets to USD for IsaacLab (M0 of docs/isaaclab_migration.md).
#
# Thin wrapper around IsaacLab's own scripts/tools/convert_urdf.py so there is no
# converter of ours to keep in step with IsaacLab. That script launches a fresh Isaac Sim
# per asset, so this takes roughly an hour for the ~96 assets -- it is a one-time step and
# the output is gitignored.
#
# This only works because assets/allegro/*.urdf were fixed at the source: USD prim paths
# cannot contain '.', and the importer resolves mesh paths relative to the URDF rather
# than relative to assets/. See "M0 findings" in docs/isaaclab_migration.md.
#
# Usage:
#   scripts/convert_assets.sh                      # convert everything missing
#   scripts/convert_assets.sh --force              # re-convert everything
#   ISAACLAB=~/path/to/IsaacLab scripts/convert_assets.sh
set -uo pipefail

# Everything lives inside main() so bash parses the whole script before running any of
# it. Bash otherwise reads scripts lazily by byte offset, and editing this file while a
# ~1 hour run is in progress makes the interpreter resume mid-construct and die with a
# spurious syntax error.
main() {
    local script_dir repo_root conv assets out force ok skip fail
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(dirname "$script_dir")"

    ISAACLAB="${ISAACLAB:-$HOME/code/IsaacLab}"
    conv="$ISAACLAB/scripts/tools/convert_urdf.py"
    assets="$repo_root/assets"
    out="$assets/usd"
    force=0
    [ "${1:-}" = "--force" ] && force=1

    [ -f "$conv" ] || { echo "IsaacLab converter not found at $conv (set ISAACLAB=)"; return 1; }
    [ -d "$assets" ] || { echo "assets/ not found next to $script_dir"; return 1; }

    ok=0; skip=0; fail=0
    local failed=() urdfs=()
    mapfile -t urdfs < <(find "$assets" -name '*.urdf' -not -path "$out/*" | sort)
    echo "converting ${#urdfs[@]} asset(s) -> $out"

    local urdf rel usd args
    for urdf in "${urdfs[@]}"; do
        rel="${urdf#"$assets"/}"
        usd="$out/${rel%.urdf}.usd"
        if [ -f "$usd" ] && [ $force -eq 0 ]; then
            skip=$((skip+1))
            echo "[$((ok+skip+fail))/${#urdfs[@]}] $rel (already converted)"
            continue
        fi
        mkdir -p "$(dirname "$usd")"

        # The hands are fixed-base articulations driven by hora's own PD loop, so the
        # baked drive gains are zeroed (target_type none) to avoid PhysX also controlling
        # the joints. Everything else in the tree is a single free rigid body.
        args=(--headless --merge-joints)
        case "$rel" in
            allegro/*) args+=(--fix-base --joint-stiffness 0 --joint-damping 0 --joint-target-type none) ;;
        esac

        echo "[$((ok+skip+fail+1))/${#urdfs[@]}] $rel"
        if python "$conv" "$urdf" "$usd" "${args[@]}" > /dev/null 2>&1 && [ -f "$usd" ]; then
            ok=$((ok+1))
        else
            fail=$((fail+1)); failed+=("$rel")
            echo "    FAILED"
        fi
    done

    echo "----------------------------------------------------------------------"
    echo "converted: $ok   skipped: $skip   failed: $fail"
    if [ $fail -gt 0 ]; then
        printf '  %s\n' "${failed[@]}"
        return 1
    fi
    echo "next: python scripts/verify_hand_asset.py \\"
    echo "          assets/usd/allegro/allegro_internal.usd assets/allegro/allegro_internal.urdf"
}

main "$@"

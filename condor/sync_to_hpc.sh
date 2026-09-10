#!/usr/bin/env bash
#
# Sync gitignored project data with the SIC HTCondor submit node. Upload mode
# sends the inputs a `git pull` cannot carry; --from-hpc retrieves job outputs.
#
# /home is NFS-shared with every worker node, so this only has to happen once.
#
# Usage:
#   condor/sync_to_hpc.sh <sic-user> [extra-path ...]
#   condor/sync_to_hpc.sh <sic-user> --only <path> [path ...] [--delete] [--dry-run]
#   condor/sync_to_hpc.sh <sic-user> --from-hpc [path ...]
#   condor/sync_to_hpc.sh <sic-user> --from-hpc --only <path> [path ...] [--delete] [--dry-run]
#
#   --from-hpc   reverse direction and download generated outputs. Defaults to
#                checkpoints, evaluations, artifacts, and condor/logs.
#   extra-path   additional file or folder, relative to the repo root. Its path
#                is recreated at the destination.
#   --only       sync just the given paths, skipping the default set above.
#   --delete     remove destination files absent at the source. Requires --only;
#                pair it with --dry-run first.
#   --dry-run    list what would change without touching the cluster.
#   REMOTE_DIR   env var overriding the repo path on the cluster.
#
# Re-running is cheap: rsync skips files that already match.

set -euo pipefail

HOST=conduit2.hpc.uni-saarland.de
TARGET="${1:?usage: condor/sync_to_hpc.sh <sic-user> [extra-path ...]}@$HOST"
REMOTE_DIR="${REMOTE_DIR:-alzheimer-detection}"
shift

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ONLY=0
DELETE=0
DIRECTION=to_hpc
RSYNC=(-avPR)   # -R recreates each source path under the destination
GIVEN=()

for arg in "$@"; do
    case "$arg" in
        --only)    ONLY=1 ;;
        --delete)  DELETE=1; RSYNC+=(--delete) ;;
        --dry-run) RSYNC+=(--dry-run) ;;
        --from-hpc) DIRECTION=from_hpc ;;
        --*)       echo "unknown flag: $arg" >&2; exit 2 ;;
        *)         GIVEN+=("$arg") ;;
    esac
done

if (( ONLY )); then
    PATHS=("${GIVEN[@]:?--only needs at least one path}")
elif (( DELETE )); then
    echo "error: --delete requires --only, to keep deletion aimed at one explicit path" >&2
    exit 2
else
    if [[ "$DIRECTION" == from_hpc ]]; then
        PATHS=(checkpoints evaluations artifacts condor/logs)
        # A fresh cluster checkout may not have every output directory yet.
        RSYNC+=(--ignore-missing-args)
    else
        PATHS=(
            data/T88_111_masked
            data/oasis_cross-sectional_cdr_cleaned.xlsx
            configs
            pretrained
        )

        # train.py load_dotenv()s this for WANDB_API_KEY. Absent locally means the
        # cluster can only run wandb offline, so say so rather than fail silently.
        if [[ -e .env ]]; then
            PATHS+=(.env)
        else
            echo "warning: no .env -- WANDB_API_KEY won't reach the cluster" >&2
        fi
    fi

    PATHS+=("${GIVEN[@]}")
fi

if [[ "$DIRECTION" == from_hpc ]]; then
    RSYNC+=(--partial)
    # Remote paths pass through a shell. Limit pull targets to ordinary
    # repository-relative paths, which also prevents writing outside this repo.
    for path in "${PATHS[@]}"; do
        if [[ -z "$path" || "$path" == /* || "$path" == "." || "$path" == ".." || "$path" == ../* || "$path" == */../* || "$path" == */.. || ! "$path" =~ ^[A-Za-z0-9._/-]+$ ]]; then
            echo "error: pull path must stay inside the repository: '$path'" >&2
            exit 2
        fi
    done
    if [[ ! "$REMOTE_DIR" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "error: REMOTE_DIR contains unsupported characters" >&2
        exit 2
    fi
fi

# One reused connection, so you authenticate once instead of three times.
# The socket lives in a temp dir, so this leaves nothing behind in ~/.ssh.
SOCKET_DIR="$(mktemp -d)"
SOCKET="$SOCKET_DIR/cm"
SSH=(ssh -o ControlPath="$SOCKET")
cleanup() {
    "${SSH[@]}" -O exit "$TARGET" 2>/dev/null || true
    rmdir "$SOCKET_DIR" 2>/dev/null || true
}
trap cleanup EXIT
"${SSH[@]}" -o ControlMaster=yes -o ControlPersist=10m -fN "$TARGET"

if [[ "$DIRECTION" == from_hpc ]]; then
    # /./ is rsync's relative-path anchor: only the portion after it is recreated
    # locally, rather than REMOTE_DIR itself.
    for path in "${PATHS[@]}"; do
        rsync "${RSYNC[@]}" -e "ssh -o ControlPath=$SOCKET" \
            "$TARGET:$REMOTE_DIR/./${path%/}" ./
    done
    echo "done: synced ${PATHS[*]} from $TARGET:$REMOTE_DIR"
else
    "${SSH[@]}" "$TARGET" "mkdir -p '$REMOTE_DIR'"

    rsync "${RSYNC[@]}" -e "ssh -o ControlPath=$SOCKET" \
        "${PATHS[@]}" "$TARGET:$REMOTE_DIR/"

    # Reports the cluster's state, not just this run's, so it stays meaningful
    # after an --only sync that didn't touch the volumes.
    echo "done: $("${SSH[@]}" "$TARGET" "ls '$REMOTE_DIR/data/T88_111_masked' 2>/dev/null | wc -l") volumes on the cluster (expect 470)"
fi

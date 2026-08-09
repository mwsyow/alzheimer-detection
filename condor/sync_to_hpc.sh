#!/usr/bin/env bash
#
# Migrate everything git can't carry to the SIC HTCondor submit node: the masked
# OASIS volumes, the cleaned labels, .env, configs, and pretrained weights are
# all gitignored, so a `git pull` on the cluster leaves them behind.
#
# /home is NFS-shared with every worker node, so this only has to happen once.
#
# Usage:
#   condor/sync_to_hpc.sh <sic-user> [extra-path ...]
#   condor/sync_to_hpc.sh <sic-user> --only <path> [path ...] [--delete] [--dry-run]
#
#   extra-path   any additional file or folder, relative to the repo root.
#                Its path is recreated on the remote, so `notebooks/` lands in
#                <remote-dir>/notebooks/.
#   --only       sync just the given paths, skipping the default set above.
#   --delete     remove remote files that no longer exist locally. Requires
#                --only: with the default set, several sources share an implied
#                parent (data/) and deletion around implied dirs is too subtle
#                to aim safely. Pair it with --dry-run first.
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
RSYNC=(-avPR)   # -R recreates each source path under the destination
GIVEN=()

for arg in "$@"; do
    case "$arg" in
        --only)    ONLY=1 ;;
        --delete)  DELETE=1; RSYNC+=(--delete) ;;
        --dry-run) RSYNC+=(--dry-run) ;;
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

    PATHS+=("${GIVEN[@]}")
fi

# One reused connection, so you authenticate once instead of three times.
# The socket lives in a temp dir, so this leaves nothing behind in ~/.ssh.
SOCKET="$(mktemp -d)/cm"
SSH=(ssh -o ControlPath="$SOCKET")
trap '"${SSH[@]}" -O exit "$TARGET" 2>/dev/null || true' EXIT
"${SSH[@]}" -o ControlMaster=yes -o ControlPersist=10m -fN "$TARGET"

"${SSH[@]}" "$TARGET" "mkdir -p '$REMOTE_DIR'"

rsync "${RSYNC[@]}" -e "ssh -o ControlPath=$SOCKET" \
    "${PATHS[@]}" "$TARGET:$REMOTE_DIR/"

# Reports the cluster's state, not just this run's, so it stays meaningful
# after an --only sync that didn't touch the volumes.
echo "done: $("${SSH[@]}" "$TARGET" "ls '$REMOTE_DIR/data/T88_111_masked' 2>/dev/null | wc -l") volumes on the cluster (expect 470)"

#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="/media/volume/aquillmdev2-vol/docker-data"
SOURCE_ROOT="/var/lib/docker"
DAEMON_JSON="/etc/docker/daemon.json"
DRY_RUN=0
ASSUME_YES=0
KEEP_OLD_LINK=0

usage() {
  cat <<'EOF'
Migrate Docker data-root to a new disk path (non-destructive by default).

Usage:
  migrate_docker_data_root.sh [options]

Options:
  --target <path>         New Docker data-root (default: /media/volume/aquillmdev2-vol/docker-data)
  --source <path>         Current Docker root (default: /var/lib/docker)
  --daemon-json <path>    Docker daemon config path (default: /etc/docker/daemon.json)
  --yes                   Do not prompt for confirmation
  --dry-run               Print planned actions without changing anything
  --keep-old-link         Keep symlink from old path to backup directory after migration
  -h, --help              Show this help

Notes:
  - The script keeps a timestamped backup of the old Docker root.
  - It updates daemon.json by merging/setting "data-root".
  - It does NOT delete old data; remove backup manually after validation.
EOF
}

log() {
  printf '[docker-migrate] %s\n' "$*"
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
    return 0
  fi
  eval "$@"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

confirm() {
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    return 0
  fi
  read -r -p "Continue with Docker data-root migration? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET_ROOT="${2:?missing value for --target}"
      shift 2
      ;;
    --source)
      SOURCE_ROOT="${2:?missing value for --source}"
      shift 2
      ;;
    --daemon-json)
      DAEMON_JSON="${2:?missing value for --daemon-json}"
      shift 2
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --keep-old-link)
      KEEP_OLD_LINK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

need_cmd systemctl
need_cmd rsync
need_cmd python3

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source Docker root does not exist: $SOURCE_ROOT" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_root="${SOURCE_ROOT}.bak-${timestamp}"

log "Source root : $SOURCE_ROOT"
log "Target root : $TARGET_ROOT"
log "Backup path : $backup_root"
log "Config file : $DAEMON_JSON"

if ! confirm; then
  echo "Cancelled."
  exit 1
fi

run "mkdir -p \"$TARGET_ROOT\""
run "chmod 711 \"$TARGET_ROOT\""

log "Stopping Docker services..."
run "systemctl stop docker || true"
run "systemctl stop docker.socket || true"

log "Copying Docker data to target with rsync..."
run "rsync -aHAX --numeric-ids --info=progress2 \"$SOURCE_ROOT\"/ \"$TARGET_ROOT\"/"

log "Backing up current source root..."
run "mv \"$SOURCE_ROOT\" \"$backup_root\""
run "mkdir -p \"$SOURCE_ROOT\""

if [[ "$KEEP_OLD_LINK" -eq 1 ]]; then
  log "Creating compatibility symlink at old path to backup."
  run "rmdir \"$SOURCE_ROOT\""
  run "ln -s \"$backup_root\" \"$SOURCE_ROOT\""
fi

log "Updating daemon.json data-root..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '[dry-run] merge "data-root":"%s" into %s\n' "$TARGET_ROOT" "$DAEMON_JSON"
else
  python3 - "$DAEMON_JSON" "$TARGET_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

daemon_path = Path(sys.argv[1])
target_root = sys.argv[2]

data = {}
if daemon_path.exists() and daemon_path.stat().st_size > 0:
    with daemon_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

if not isinstance(data, dict):
    raise SystemExit(f"{daemon_path} must contain a JSON object")

data["data-root"] = target_root
daemon_path.parent.mkdir(parents=True, exist_ok=True)
with daemon_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY
fi

log "Starting Docker..."
run "systemctl daemon-reload"
run "systemctl start docker"

if [[ "$DRY_RUN" -eq 0 ]]; then
  actual_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  if [[ "$actual_root" != "$TARGET_ROOT" ]]; then
    echo "Migration completed, but verification failed." >&2
    echo "Expected Docker Root Dir: $TARGET_ROOT" >&2
    echo "Actual   Docker Root Dir: ${actual_root:-<unavailable>}" >&2
    echo "Rollback commands:" >&2
    cat >&2 <<EOF
sudo systemctl stop docker
sudo cp "$DAEMON_JSON" "${DAEMON_JSON}.failed-${timestamp}"
# Edit $DAEMON_JSON and restore previous data-root or remove "data-root"
sudo rm -rf "$SOURCE_ROOT"
sudo mv "$backup_root" "$SOURCE_ROOT"
sudo systemctl start docker
EOF
    exit 2
  fi
fi

log "Migration complete."
log "Verify with: docker info | grep 'Docker Root Dir'"
log "Backup retained at: $backup_root"
log "After validation, you can reclaim space by deleting backup:"
log "  sudo rm -rf \"$backup_root\""

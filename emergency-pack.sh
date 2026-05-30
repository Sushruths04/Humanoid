#!/usr/bin/env bash
# Pack all Isaac Lab work for emergency transfer before machine shutdown.
# Usage: bash emergency-pack.sh

set -euo pipefail

WORKSPACE="${HOME}/isaaclab-workspace"
EXPORT_DIR="${WORKSPACE}/exports"
STAGING="${EXPORT_DIR}/.staging-$$"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="${EXPORT_DIR}/emergency-backup-${TIMESTAMP}.tar.gz"
MANIFEST="${EXPORT_DIR}/MANIFEST-${TIMESTAMP}.txt"

mkdir -p "${STAGING}" "${EXPORT_DIR}"

echo "=== Emergency pack started at $(date) ===" | tee "${MANIFEST}"
echo "Workspace: ${WORKSPACE}" | tee -a "${MANIFEST}"
echo "" | tee -a "${MANIFEST}"

# --- 1. Your project (code) ---
if [[ -d "${WORKSPACE}/my-humanoid-project" ]]; then
  echo "[1/6] Packing my-humanoid-project..." | tee -a "${MANIFEST}"
  rsync -a --exclude='.git' --exclude='__pycache__' --exclude='isaaclab-env' \
    "${WORKSPACE}/my-humanoid-project/" "${STAGING}/my-humanoid-project/"
fi

# --- 2. Host-side Isaac Lab logs (RSL-RL, SKRL, etc.) ---
if [[ -d "${WORKSPACE}/IsaacLab/logs" ]]; then
  echo "[2/6] Packing IsaacLab/logs..." | tee -a "${MANIFEST}"
  mkdir -p "${STAGING}/IsaacLab"
  rsync -a "${WORKSPACE}/IsaacLab/logs/" "${STAGING}/IsaacLab/logs/" 2>/dev/null || true
fi

# --- 3. Shared checkpoints/logs folders ---
for dir in checkpoints logs; do
  if [[ -d "${WORKSPACE}/${dir}" ]] && [[ -n "$(ls -A "${WORKSPACE}/${dir}" 2>/dev/null)" ]]; then
    echo "[3/6] Packing ${dir}/..." | tee -a "${MANIFEST}"
    rsync -a "${WORKSPACE}/${dir}/" "${STAGING}/${dir}/"
  fi
done

# --- 4. Find stray checkpoints anywhere in workspace ---
echo "[4/6] Finding all .pt / .pth / .ckpt files..." | tee -a "${MANIFEST}"
mkdir -p "${STAGING}/all-checkpoints"
while IFS= read -r -d '' f; do
  rel="${f#${WORKSPACE}/}"
  dest="${STAGING}/all-checkpoints/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp -a "${f}" "${dest}"
  echo "  checkpoint: ${rel} ($(du -h "${f}" | cut -f1))" >> "${MANIFEST}"
done < <(find "${WORKSPACE}" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
  ! -path '*/.git/*' ! -path '*/exports/*' ! -path '*/.staging-*/*' -print0 2>/dev/null)

# --- 5. Export Docker volumes (if container was used) ---
echo "[5/6] Exporting Docker volumes (if any)..." | tee -a "${MANIFEST}"
for vol in isaac-lab-logs isaac-lab-data; do
  if docker volume inspect "${vol}" &>/dev/null; then
    echo "  Exporting volume: ${vol}" | tee -a "${MANIFEST}"
    docker run --rm \
      -v "${vol}:/volume:ro" \
      -v "${STAGING}/docker-volumes:/backup" \
      alpine sh -c "mkdir -p /backup/${vol} && cp -a /volume/. /backup/${vol}/" 2>/dev/null || \
      echo "  WARNING: Could not export ${vol}" | tee -a "${MANIFEST}"
  else
    echo "  Volume ${vol} not found (skipped)" | tee -a "${MANIFEST}"
  fi
done

# --- 6. Setup docs (small, useful on new machine) ---
for f in SETUP-STATUS.md README-SETUP.md EMERGENCY-TRANSFER.md; do
  [[ -f "${WORKSPACE}/${f}" ]] && cp "${WORKSPACE}/${f}" "${STAGING}/"
done

# --- Create archive ---
echo "[6/6] Creating compressed archive (may take several minutes)..." | tee -a "${MANIFEST}"
tar -czf "${ARCHIVE}" -C "${STAGING}" .
rm -rf "${STAGING}"

SIZE=$(du -h "${ARCHIVE}" | cut -f1)
echo "" | tee -a "${MANIFEST}"
echo "=== DONE ===" | tee -a "${MANIFEST}"
echo "Archive: ${ARCHIVE}" | tee -a "${MANIFEST}"
echo "Size:    ${SIZE}" | tee -a "${MANIFEST}"
echo "Manifest: ${MANIFEST}" | tee -a "${MANIFEST}"
echo "" | tee -a "${MANIFEST}"
echo "NEXT: Upload the .tar.gz file before this machine shuts down!"
echo "  huggingface-cli upload USER/REPO ${ARCHIVE}"
echo "  aws s3 cp ${ARCHIVE} s3://YOUR-BUCKET/"
echo "  scp ${ARCHIVE} user@other-host:~/backups/"

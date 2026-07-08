#!/usr/bin/env bash
# Build and deploy the granite-speech-demo frontend + backend to OpenShift via
# on-cluster binary builds. Idempotent: safe to re-run to ship changes.
#
# Prereqs:
#   - oc logged into the target cluster
#   - HF_TOKEN available (env var, or uncommented in ./.env) — needed so WebRTC
#     media relays through TURN for users behind NAT
#
# Usage:
#   ./openshift/deploy.sh
set -euo pipefail

NAMESPACE="granite-speech-demo"

# Repo root, regardless of where the script is called from.
cd "$(dirname "$0")/.."

CONFIG_EXAMPLE="openshift/config.example.env"
CONFIG_FILE="openshift/config.env"

echo "==> Cluster: $(oc whoami --show-server 2>/dev/null) as $(oc whoami 2>/dev/null)"
oc project "$NAMESPACE" >/dev/null

# --- config.env (non-secret backend env) ---------------------------------
if [ ! -f "$CONFIG_FILE" ]; then
    echo "==> No $CONFIG_FILE; seeding from $CONFIG_EXAMPLE (edit it and re-run to customize)."
    cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
fi
# shellcheck disable=SC1090
set -a; source "$CONFIG_FILE"; set +a
PROMPT_SOURCE_FILE="${PROMPT_SOURCE_FILE:-prompts/granite.txt}"

# --- HF token -------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    HF_TOKEN="$(grep -E '^[[:space:]]*HF_TOKEN=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it (export HF_TOKEN=hf_...) or uncomment it in .env." >&2
    echo "       Without it, users behind NAT can't connect (no TURN relay)." >&2
    exit 1
fi

# --- Secret ---------------------------------------------------------------
echo "==> Applying Secret granite-speech-secrets"
oc create secret generic granite-speech-secrets \
    --from-literal=HF_TOKEN="$HF_TOKEN" \
    ${LLM_API_KEY:+--from-literal=LLM_API_KEY="$LLM_API_KEY"} \
    --dry-run=client -o yaml | oc apply -f -

# --- Persona ConfigMap ----------------------------------------------------
if [ ! -f "$PROMPT_SOURCE_FILE" ]; then
    echo "ERROR: persona file '$PROMPT_SOURCE_FILE' not found (set PROMPT_SOURCE_FILE in $CONFIG_FILE)." >&2
    exit 1
fi
echo "==> Applying ConfigMap granite-speech-prompt (from $PROMPT_SOURCE_FILE)"
oc create configmap granite-speech-prompt \
    --from-file=granite.txt="$PROMPT_SOURCE_FILE" \
    --dry-run=client -o yaml | oc apply -f -

# --- Config ConfigMap (drop deploy-only keys) -----------------------------
echo "==> Applying ConfigMap granite-speech-config"
TMP_ENV="$(mktemp)"
trap 'rm -f "$TMP_ENV"' EXIT
grep -vE '^[[:space:]]*(#|$)' "$CONFIG_FILE" | grep -vE '^PROMPT_SOURCE_FILE=' > "$TMP_ENV"
oc create configmap granite-speech-config \
    --from-env-file="$TMP_ENV" \
    --dry-run=client -o yaml | oc apply -f -

# --- Manifests ------------------------------------------------------------
echo "==> Applying manifests"
oc apply -f openshift/backend.yaml -f openshift/frontend.yaml

# The namespace quota (2 CPU / 4Gi) is tight. Free it for the build pods by
# parking the Deployments at 0 replicas while images build; scaled back up after.
echo "==> Scaling Deployments to 0 to free quota for build pods"
oc scale deploy/granite-speech-backend deploy/granite-speech-frontend --replicas=0 >/dev/null

# --- Staged build contexts ------------------------------------------------
# This oc client doesn't reliably honor .dockerignore for `--from-dir`, so it
# would upload the whole repo (.venv, node_modules — >1GB) on every build:
# slow, and enough scratch data to trip the build pod's ephemeral limit. Stage
# a minimal context (only what each Dockerfile needs) so uploads are a few 100KB.
STAGE="$(mktemp -d)"
trap 'rm -f "$TMP_ENV"; rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/backend" "$STAGE/frontend"
cp Dockerfile.backend pyproject.toml uv.lock README.md "$STAGE/backend/"
cp -R src prompts docs "$STAGE/backend/"
rsync -a --exclude node_modules --exclude .next --exclude .turbo \
      --exclude '*.tsbuildinfo' frontend/ "$STAGE/frontend/"

# --- Builds (on-cluster, in parallel) -------------------------------------
# Fire both without --wait (survives a dropped client connection), then poll
# the build objects to terminal state.
echo "==> Starting backend + frontend builds"
oc start-build granite-speech-backend  --from-dir="$STAGE/backend"  >/dev/null
oc start-build granite-speech-frontend --from-dir="$STAGE/frontend" >/dev/null

echo "==> Waiting for builds to finish"
for bc in granite-speech-backend granite-speech-frontend; do
    b="$(oc get build -l buildconfig="$bc" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
    while :; do
        phase="$(oc get build "$b" -o jsonpath='{.status.phase}')"
        case "$phase" in
            Complete) echo "    $b: Complete"; break ;;
            Failed|Error|Cancelled) echo "    $b: $phase"; oc logs "build/$b" 2>/dev/null | tail -20; exit 1 ;;
            *) sleep 10 ;;
        esac
    done
done

# --- Rollout --------------------------------------------------------------
echo "==> Scaling Deployments back up"
oc scale deploy/granite-speech-backend deploy/granite-speech-frontend --replicas=1 >/dev/null
echo "==> Waiting for rollouts"
oc rollout status deploy/granite-speech-backend --timeout=300s
oc rollout status deploy/granite-speech-frontend --timeout=300s

# --- Done -----------------------------------------------------------------
HOST="$(oc get route granite-speech-frontend -o jsonpath='{.spec.host}')"
echo
echo "======================================================================"
echo " Deployed. Share this URL:"
echo "   https://$HOST"
echo "======================================================================"

#!/usr/bin/env bash
# Run an image's own config generator on the Fikar production configs (KAN-606).
#
#   tests/fikar/check-configs.sh <image>
#
# The fixtures are copies of config-pinned.yaml and config-routed.yaml from Kapture-AI/plano-gateway,
# keys left as $VARS. Each one is mounted where the gateway mounts it (/app/plano_config.yaml) and
# planoai.config_generator validates it against the schema baked into the image, the same step
# supervisord runs before envoy and brightstaff start. Any failure (a field the schema rejects, such
# as tracing.attribution_header on an image without the Fikar patches) exits 1. No network, no keys.
set -euo pipefail

IMAGE="${1:?usage: check-configs.sh <image>}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

failed=()
for cfg in "$DIR"/config-*.yaml; do
  name="$(basename "$cfg")"
  echo "::group::$name on $IMAGE"
  # python, not "uv run": the image installs planoai system-wide, and uv would try to build a venv
  # from the network.
  if docker run --rm --network none --entrypoint sh \
      -v "$cfg:/app/plano_config.yaml:ro" "$IMAGE" \
      -c 'python -m planoai.config_generator > /tmp/out 2>&1 || { cat /tmp/out; exit 1; }'; then
    echo "ok: $name"
  else
    failed+=("$name")
  fi
  echo "::endgroup::"
done

if ((${#failed[@]})); then
  for name in "${failed[@]}"; do echo "::error title=Plano config rejected::$name fails config generation on $IMAGE"; done
  exit 1
fi
echo "all configs accepted by $IMAGE"

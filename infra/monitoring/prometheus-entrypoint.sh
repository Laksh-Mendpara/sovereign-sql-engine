#!/bin/sh
set -eu

CONFIG_IN="/etc/prometheus/prometheus.yml"
CONFIG_OUT="/tmp/prometheus.yml"

cp "$CONFIG_IN" "$CONFIG_OUT"

if [ -n "${GRAFANA_CLOUD_PROM_REMOTE_WRITE_URL:-}" ] && \
   [ -n "${GRAFANA_CLOUD_PROM_USERNAME:-}" ] && \
   [ -n "${GRAFANA_CLOUD_PROM_API_KEY:-}" ]; then
  PROM_BASE_URL="${GRAFANA_CLOUD_PROM_REMOTE_WRITE_URL%/}"

  case "$PROM_BASE_URL" in
    */api/prom/push)
      PROM_WRITE_URL="$PROM_BASE_URL"
      ;;
    */api/prom)
      PROM_WRITE_URL="$PROM_BASE_URL/push"
      ;;
    *)
      PROM_WRITE_URL="$PROM_BASE_URL/api/prom/push"
      ;;
  esac

  cat <<EOF >> "$CONFIG_OUT"

remote_write:
  - url: ${PROM_WRITE_URL}
    basic_auth:
      username: ${GRAFANA_CLOUD_PROM_USERNAME}
      password: ${GRAFANA_CLOUD_PROM_API_KEY}
EOF
fi

exec /bin/prometheus \
  --config.file="$CONFIG_OUT" \
  --storage.tsdb.path=/prometheus \
  --web.enable-lifecycle \
  --web.listen-address=0.0.0.0:9090

#!/bin/sh
set -eu

if [ -z "${GRAFANA_CLOUD_LOKI_URL:-}" ] || \
   [ -z "${GRAFANA_CLOUD_LOKI_USERNAME:-}" ] || \
   [ -z "${GRAFANA_CLOUD_LOKI_API_KEY:-}" ]; then
  echo "Grafana Cloud Loki env vars are not set; promtail will idle without shipping logs."
  tail -f /dev/null
fi

case "$GRAFANA_CLOUD_LOKI_URL" in
  */loki/api/v1/push)
    export GRAFANA_CLOUD_LOKI_URL="${GRAFANA_CLOUD_LOKI_URL%/}"
    ;;
  */loki/api/v1)
    export GRAFANA_CLOUD_LOKI_URL="${GRAFANA_CLOUD_LOKI_URL%/}/push"
    ;;
  */loki)
    export GRAFANA_CLOUD_LOKI_URL="${GRAFANA_CLOUD_LOKI_URL%/}/api/v1/push"
    ;;
  *)
    export GRAFANA_CLOUD_LOKI_URL="${GRAFANA_CLOUD_LOKI_URL%/}/loki/api/v1/push"
    ;;
esac

exec /usr/bin/promtail \
  -config.file=/etc/promtail/promtail.yml \
  -config.expand-env=true

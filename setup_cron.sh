#!/usr/bin/env bash
#
# Install (or refresh) the cron job that runs the Heston paper-trading system
# once per trading day. Idempotent: re-running replaces the existing entry
# rather than adding a duplicate.
#
# Prerequisites at run time (this installer does not check or start them):
#   - TWS / IB Gateway running and logged into paper account DUR195917, with
#     the API enabled on 127.0.0.1:7497;
#   - the project virtualenv at .venv with dependencies installed;
#   - a Gmail app password in ~/.heston_secrets for the email report.
#
# NOTE: cron fires in the machine's LOCAL timezone. The default schedule below
# assumes local time is London (BST); adjust CRON_SCHEDULE if it is not.
set -euo pipefail

# --- Configuration ----------------------------------------------------------------
readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
readonly RUNNER_MODULE="src.paper_trading.runner"
readonly LOG_DIR="${PROJECT_DIR}/logs"
readonly CRON_LOG="${LOG_DIR}/cron.log"
# Weekdays (Mon-Fri) at 21:00 London (BST) = 16:00 ET, the cash close.
readonly CRON_SCHEDULE="0 21 * * 1-5"
# Unique marker comment so re-running this script replaces our entry only.
readonly CRON_MARKER="# heston-paper-trading"

readonly CRON_COMMAND="cd ${PROJECT_DIR} && ${VENV_PYTHON} -m ${RUNNER_MODULE} >> ${CRON_LOG} 2>&1"
readonly CRON_LINE="${CRON_SCHEDULE} ${CRON_COMMAND} ${CRON_MARKER}"

main() {
    if [[ ! -x "${VENV_PYTHON}" ]]; then
        echo "error: no executable virtualenv python at ${VENV_PYTHON}" >&2
        exit 1
    fi
    mkdir -p "${LOG_DIR}"

    # Drop any prior entry carrying our marker, then append the fresh line.
    local kept
    kept="$(crontab -l 2>/dev/null | grep -v -F "${CRON_MARKER}" || true)"
    printf '%s\n%s\n' "${kept}" "${CRON_LINE}" | grep -v '^[[:space:]]*$' | crontab -

    echo "installed cron job:"
    echo "  ${CRON_LINE}"
    echo "cron output logs to ${CRON_LOG}"
}

main "$@"

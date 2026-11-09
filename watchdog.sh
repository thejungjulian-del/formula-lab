#!/bin/sh
# Restarts the Formula Lab server if it doesn't answer within 10 seconds (runs every 2 minutes via launchd).
if ! /usr/bin/curl -sf -m 10 -o /dev/null http://127.0.0.1:8802/api/ping; then
  echo "$(date) server not answering, restarting"
  /bin/launchctl kickstart -k gui/$(id -u)/com.formulalab.server
fi

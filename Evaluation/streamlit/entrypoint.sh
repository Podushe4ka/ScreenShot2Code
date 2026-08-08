#!/usr/bin/env bash
# entrypoint.sh — запускает streamlit, прокидывая ВСЕ аргументы контейнера
# (docker run ... IMAGE --model ... --device cuda) как аргументы app.py
# через "--" (см. app.py: argparse.parse_known_args(sys.argv[1:])).
set -euo pipefail

exec streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    -- \
    "$@"

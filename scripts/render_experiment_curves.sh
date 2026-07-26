#!/usr/bin/env bash

set -euo pipefail

repo_dir="/home/kedong/projects/cs336/assignments/assignment1-basics"
cd "$repo_dir"
mkdir -p reports/assets

render_svg() {
    local output_path="$1"
    local title="$2"
    local x_min="$3"
    local x_max="$4"
    local y_min="$5"
    local y_max="$6"
    shift 6

    local width=1120
    local height=600
    local left=80
    local top=60
    local plot_width=710
    local plot_height=460
    local right=$((left + plot_width))
    local bottom=$((top + plot_height))

    {
        echo '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="600" viewBox="0 0 1120 600" role="img">'
        echo "<title>$title</title>"
        echo '<rect width="1120" height="600" fill="#ffffff"/>'
        echo '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#24292f}.grid{stroke:#d8dee4;stroke-width:1}.axis{stroke:#57606a;stroke-width:1.5}.curve{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}.tick{font-size:13px}.legend{font-size:14px}.title{font-size:20px;font-weight:600}.label{font-size:15px;font-weight:500}</style>'
        echo "<text class=\"title\" x=\"$((left + plot_width / 2))\" y=\"32\" text-anchor=\"middle\">$title</text>"

        for tick in 0 1 2 3 4 5; do
            local x
            local value
            x="$(awk -v l="$left" -v w="$plot_width" -v i="$tick" 'BEGIN { printf "%.2f", l + w * i / 5 }')"
            value="$(awk -v a="$x_min" -v b="$x_max" -v i="$tick" 'BEGIN { printf "%.0f", a + (b-a) * i / 5 }')"
            echo "<line class=\"grid\" x1=\"$x\" y1=\"$top\" x2=\"$x\" y2=\"$bottom\"/>"
            echo "<text class=\"tick\" x=\"$x\" y=\"$((bottom + 24))\" text-anchor=\"middle\">$value</text>"
        done

        for tick in 0 1 2 3 4 5 6 7 8; do
            local y
            local value
            y="$(awk -v t="$top" -v h="$plot_height" -v i="$tick" 'BEGIN { printf "%.2f", t + h * i / 8 }')"
            value="$(awk -v a="$y_min" -v b="$y_max" -v i="$tick" 'BEGIN { printf "%.2f", b - (b-a) * i / 8 }')"
            echo "<line class=\"grid\" x1=\"$left\" y1=\"$y\" x2=\"$right\" y2=\"$y\"/>"
            echo "<text class=\"tick\" x=\"$((left - 12))\" y=\"$(awk -v y="$y" 'BEGIN { printf "%.2f", y + 4 }')\" text-anchor=\"end\">$value</text>"
        done

        echo "<line class=\"axis\" x1=\"$left\" y1=\"$bottom\" x2=\"$right\" y2=\"$bottom\"/>"
        echo "<line class=\"axis\" x1=\"$left\" y1=\"$top\" x2=\"$left\" y2=\"$bottom\"/>"
        echo "<text class=\"label\" x=\"$((left + plot_width / 2))\" y=\"575\" text-anchor=\"middle\">optimizer step</text>"
        echo "<text class=\"label\" transform=\"translate(20 $((top + plot_height / 2))) rotate(-90)\" text-anchor=\"middle\">validation loss (per token)</text>"

        local legend_y=82
        local spec
        for spec in "$@"; do
            local label="${spec%%|*}"
            local rest="${spec#*|}"
            local color="${rest%%|*}"
            local csv_path="${rest#*|}"
            local points
            points="$(
                awk -F, \
                    -v xmin="$x_min" -v xmax="$x_max" \
                    -v ymin="$y_min" -v ymax="$y_max" \
                    -v left="$left" -v top="$top" \
                    -v width="$plot_width" -v height="$plot_height" '
                    NR > 1 && $1 + 0 >= xmin && $1 + 0 <= xmax {
                        x = left + (($1 - xmin) / (xmax - xmin)) * width
                        y = top + ((ymax - $2) / (ymax - ymin)) * height
                        if (y < top) y = top
                        if (y > top + height) y = top + height
                        printf "%.2f,%.2f ", x, y
                    }
                ' "$csv_path"
            )"
            echo "<polyline class=\"curve\" stroke=\"$color\" points=\"$points\"/>"
            echo "<line x1=\"825\" y1=\"$legend_y\" x2=\"855\" y2=\"$legend_y\" stroke=\"$color\" stroke-width=\"3\"/>"
            echo "<text class=\"legend\" x=\"866\" y=\"$((legend_y + 5))\">$label</text>"
            legend_y=$((legend_y + 26))
        done
        echo '</svg>'
    } > "$output_path"
}

render_svg \
    reports/assets/learning_rate_sweep.svg \
    "Learning-rate sweep (batch 128, 3000 steps)" \
    0 3000 1.5 9.5 \
    "lr=3e-4|#cf222e|reports/data/lr_b128_3e-4_s3000_val.csv" \
    "lr=1e-3|#0969da|reports/data/lr_b128_1e-3_s3000_val.csv" \
    "lr=3e-3|#1a7f37|reports/data/lr_b128_3e-3_s3000_val.csv"

render_svg \
    reports/assets/learning_rate_10k.svg \
    "Long-run validation loss after resuming at step 3000" \
    3000 10000 1.70 2.20 \
    "lr=1e-3|#0969da|reports/data/lr_b128_1e-3_s10000_resume_val.csv" \
    "lr=3e-3|#1a7f37|reports/data/lr_b128_3e-3_s10000_resume_val.csv"

render_svg \
    reports/assets/batch_lr_sweeps.svg \
    "Batch-size learning-rate sweeps (1000 steps)" \
    0 1000 2.4 9.5 \
    "b=1, lr=2.34375e-5|#8250df|reports/data/overnight_batch1_lr2.34375e-5_s1000_val.csv" \
    "b=32, lr=3.75e-4|#54aeff|reports/data/overnight_batch32_lr3.75e-4_s1000_val.csv" \
    "b=32, lr=7.5e-4|#0969da|reports/data/overnight_batch32_lr7.5e-4_s1000_val.csv" \
    "b=32, lr=1.5e-3|#033d8b|reports/data/overnight_batch32_lr1.5e-3_s1000_val.csv" \
    "b=64, lr=7.5e-4|#7ee787|reports/data/overnight_batch64_lr7.5e-4_s1000_val.csv" \
    "b=64, lr=1.5e-3|#1a7f37|reports/data/overnight_batch64_lr1.5e-3_s1000_val.csv" \
    "b=64, lr=3e-3|#0f5323|reports/data/overnight_batch64_lr3.0e-3_s1000_val.csv"

render_svg \
    reports/assets/best_by_batch.svg \
    "Best tested learning rate for each batch size" \
    0 1000 2.2 9.5 \
    "b=1, lr=2.34375e-5|#8250df|reports/data/overnight_batch1_lr2.34375e-5_s1000_val.csv" \
    "b=32, lr=1.5e-3|#0969da|reports/data/overnight_batch32_lr1.5e-3_s1000_val.csv" \
    "b=64, lr=3e-3|#1a7f37|reports/data/overnight_batch64_lr3.0e-3_s1000_val.csv" \
    "b=128, lr=3e-3|#bc4c00|reports/data/lr_b128_3e-3_s3000_val.csv"

echo "Rendered experiment curves under reports/assets/."

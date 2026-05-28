#!/usr/bin/env zsh
set -euo pipefail

src_dir="${1:-diagrams}"
out_dir="${2:-diagrams}"
index_file="${out_dir}/index.html"

mkdir -p "$out_dir"

cat > "$index_file" <<'HTML'
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ML Mermaid Diagram Index</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }
    li { margin: 0.35rem 0; }
  </style>
</head>
<body>
  <h1>ML Mermaid Diagram Index</h1>
  <ul>
HTML

for mmd in "$src_dir"/*.mmd(N); do
  base="${mmd:t:r}"
  svg="${out_dir}/${base}.svg"

  echo "Rendering $mmd -> $svg"
  mmdc -i "$mmd" -o "$svg"

  cat >> "$index_file" <<HTML
    <li><a href="./${base}.svg">${base}.svg</a></li>
HTML
done

cat >> "$index_file" <<'HTML'
  </ul>
</body>
</html>
HTML

echo "Wrote $index_file"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <base-sha> <head-sha> <findings-output>" >&2
  exit 2
fi

BASE="$1"
HEAD="$2"
FINDINGS_OUTPUT="$3"

DIFF=$(git diff "$BASE".."$HEAD" -- . ':!uv.lock' ':!*.lock' ':!package-lock.json' ':!yarn.lock' || true)
FINDINGS=""

PTH_FILES=$(git diff --name-only "$BASE".."$HEAD" | grep '\.pth$' || true)
if [ -n "$PTH_FILES" ]; then
  FINDINGS="${FINDINGS}
### 🚨 CRITICAL: .pth file added or modified
Python \`.pth\` files in \`site-packages/\` execute automatically when the interpreter starts — no import required.

**Files:**
\`\`\`
${PTH_FILES}
\`\`\`
"
fi

B64_EXEC_HITS=$(echo "$DIFF" | grep -n '^\+' | grep -v '^\+\+\+' | grep -iE 'base64\.(b64decode|decodebytes|urlsafe_b64decode)' | grep -iE 'exec\(|eval\(' | head -10 || true)
if [ -n "$B64_EXEC_HITS" ]; then
  FINDINGS="${FINDINGS}
### 🚨 CRITICAL: base64 decode + exec/eval combo
Base64-decoded strings passed directly to exec/eval — the signature of hidden credential-stealing payloads.

**Matches:**
\`\`\`
${B64_EXEC_HITS}
\`\`\`
"
fi

PROC_HITS=$(echo "$DIFF" | grep -n '^\+' | grep -v '^\+\+\+' | grep -E 'subprocess\.(Popen|call|run)\s*\(' | grep -iE 'base64|\\x[0-9a-f]{2}|chr\(' | head -10 || true)
if [ -n "$PROC_HITS" ]; then
  FINDINGS="${FINDINGS}
### 🚨 CRITICAL: subprocess with encoded/obfuscated command
Subprocess calls whose command strings are base64- or hex-encoded are a strong indicator of payload execution.

**Matches:**
\`\`\`
${PROC_HITS}
\`\`\`
"
fi

SETUP_HITS=$(git diff --name-only "$BASE".."$HEAD" | grep -E '^(setup\.py|setup\.cfg)$|(^|/)(sitecustomize\.py|usercustomize\.py|__init__\.pth)$' || true)
if [ -n "$SETUP_HITS" ]; then
  FINDINGS="${FINDINGS}
### 🚨 CRITICAL: Install-hook file added or modified
These files can execute code during package installation or interpreter startup.

**Files:**
\`\`\`
${SETUP_HITS}
\`\`\`
"
fi

if [ -n "$FINDINGS" ]; then
  printf '%s\n' "$FINDINGS" > "$FINDINGS_OUTPUT"
  echo "found=true"
  exit 1
fi

rm -f "$FINDINGS_OUTPUT"
echo "found=false"

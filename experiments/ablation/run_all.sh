#!/usr/bin/env bash
# Reproduces the 24 ablation runs in ablation_results.jsonl (run from the repo root).
set -e
cd "$(dirname "$0")/../.."
A=experiments/ablation/ablation.py
for cfg in '{}' '{"text":"clean"}' '{"text":"raw_kw"}' '{"text":"clean_kw"}' '{"text":"clean_kw_url"}' \
  '{"rec_dropout":0.0}' '{"units":64}' '{"units":256}' '{"maxlen":35}' '{"max_words":20000}' \
  '{"spatial":0.2}' '{"finetune_epochs":3}' '{"dedupe":true}' '{"batch":32}' '{"lr":5e-4}' \
  '{"dropout":0.3,"dense":128}' \
  '{"spatial":0.2,"text":"raw_kw","max_words":20000,"epochs":25,"patience":4}' \
  '{"spatial":0.2,"text":"raw_kw","dedupe":true,"epochs":25,"patience":4}' \
  '{"spatial":0.2,"text":"raw_kw","dedupe":true,"max_words":20000,"batch":32,"epochs":25,"patience":4,"finetune_epochs":3}' \
  '{"spatial":0.3,"text":"raw_kw","units":64,"dedupe":true,"epochs":25,"patience":4,"lr":5e-4}'; do
  python $A "$cfg" 42
done
for seed in 1 2 3 4; do python $A '{}' $seed; done

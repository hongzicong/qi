# server

```bash
python scripts/serve_policy.py \
  --config configs/real_bot.yaml \
  --host 127.0.0.1 \
  --port 8765 \
  --authkey wam
```

# warmup
```bash
python scripts/deploy_real_rtc_wam.py \
    --config configs/real_bot.yaml \
    --no-execute-actions \
    --no-move-to-initial \
    --max-steps 3
```

# inference

```bash
python scripts/deploy_real_rtc_wam.py \
  --config configs/real_bot.yaml \
  --move-to-initial \
  --execute-actions
```

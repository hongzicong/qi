
```bash
PYTHONPATH=src python scripts/serve_policy.py \
  --config configs/real_bot.yaml \
  --host 127.0.0.1 \
  --port 8765 \
  --authkey wam
```

```bash
PYTHONPATH=src python scripts/deploy_real_rtc_wam.py \
  --config configs/real_bot.yaml \
  --max-steps 1 \
  --move-to-initial \
  --execute-actions \
  --arm-mode single \
  --arm-side right
```
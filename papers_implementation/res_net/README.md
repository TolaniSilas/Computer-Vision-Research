## ResNet




## training

run a sanity check first to confirm everything is working, as expected:

```bash
bash scripts/train.sh --sanity_check
```

once confirmed, run full training:

```bash
bash scripts/train.sh
```

training settings such as epochs, learning rate, and batch size can be configured in `config/default.yaml`.
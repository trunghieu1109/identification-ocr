# Architecture sources

The architecture files are copied without modification from the authors:

- `restormer.py`: [swz30/Restormer](https://github.com/swz30/Restormer/blob/68dc6ac472db26f16361150cb7a96a1bc87da93f/basicsr/models/archs/restormer_arch.py), commit `68dc6ac472db26f16361150cb7a96a1bc87da93f`. MIT license in `LICENSE.restormer`.
- `swinir.py`: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py), commit `6545850fbf8df298df73d81f3e8cba638787c8bd`. Apache 2.0 license in `LICENSE.swinir`.

Weights are downloaded separately by `scripts/setup-qr-restoration.py`:
Restormer's single-image defocus checkpoint from the authors' Google Drive,
and SwinIR's lightweight DIV2K x4 checkpoint from the authors' GitHub release.
The QR experiment integration, tiling and coordinate mapping live outside these files.

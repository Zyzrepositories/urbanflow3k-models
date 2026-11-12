# UrbanFlow-3K model implementations

The two PyTorch models used in the UrbanFlow-3K comparison are the rectangular convolutional neural operator (`cno_rect`, 999,712 trainable parameters) and U-Net (`unet`, 1,086,194 trainable parameters). This repository contains the model definitions and their common padded inference interface.

Use PyTorch 2.5.1 and NumPy. Import `build_model` from `models.py` while the repository root is on the Python import path.

```python
import torch
from models import build_model

model = build_model("cno_rect").eval()  # or "unet"
x = torch.zeros(1, 5, 140, 180)  # shape example
with torch.inference_mode():
    velocity = model(x)
assert velocity.shape == (1, 2, 140, 180)
```

The input channel order is fluid mask, clipped signed raster distance, normalized x coordinate, normalized y coordinate, and Reynolds number encoded as (Re − 4000) / 1000. The output contains the two normalized velocity components. The factory initializes fresh weights.

`_cno_upstream.py` is the CNO reference implementation from [camlab-ethz/ConvolutionalNeuralOperator](https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/6e765198aa02b56352e0a3437104b9d9e337176e), commit `6e765198aa02b56352e0a3437104b9d9e337176e`. Its MIT notice is in `LICENSE-CNO`. The rectangular grid adapter is in `models.py`.

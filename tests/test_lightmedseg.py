import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import LightMedSegOutput, lightmedseg_tiny


def test_lightmedseg_forward_shape() -> None:
    model = lightmedseg_tiny(in_channels=3, num_classes=1)
    x = torch.randn(2, 3, 128, 128)

    y = model(x)

    assert y.shape == (2, 1, 128, 128)


def test_lightmedseg_aux_output_shape() -> None:
    model = lightmedseg_tiny(in_channels=3, num_classes=1)
    x = torch.randn(2, 3, 128, 128)

    output = model(x, return_aux=True)

    assert isinstance(output, LightMedSegOutput)
    assert output.mask_logits.shape == (2, 1, 128, 128)
    assert output.edge_logits.shape == (2, 1, 128, 128)


if __name__ == "__main__":
    test_lightmedseg_forward_shape()
    test_lightmedseg_aux_output_shape()
    print("LightMedSeg smoke test passed.")

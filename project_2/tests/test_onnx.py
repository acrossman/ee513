import numpy as np

from isp.deploy import export_onnx, load_session, run_onnx
from isp.dncnn import SmallDnCNN, denoise


def test_onnx_matches_pytorch(tmp_path):
    model = SmallDnCNN()
    path = tmp_path / "model.onnx"
    export_onnx(model, path)
    img = np.random.default_rng(5).random((40, 56)).astype(np.float32)
    assert np.max(np.abs(run_onnx(load_session(path), img) - denoise(model, img))) < 1e-5

import time

import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_onnx(model, path):
    model.eval()
    example = torch.zeros(1, 1, 64, 64)
    height, width = torch.export.Dim("height", min=8), torch.export.Dim("width", min=8)
    program = torch.onnx.export(model, (example,), dynamo=True,
                                input_names=["noisy"], output_names=["denoised"],
                                dynamic_shapes={"y": {2: height, 3: width}})
    program.save(str(path))
    onnx.checker.check_model(str(path))


def load_session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(session, img):
    x = np.ascontiguousarray(img, dtype=np.float32)[None, None]
    return session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]


def time_onnx(session, img, runs=10):
    run_onnx(session, img)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_onnx(session, img)
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000

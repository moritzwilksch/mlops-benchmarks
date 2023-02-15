# MLOps Benchmarks

| Inference type | Model Format | Time for 10k (s) |
| --- | --- | --- |
| python function | joblib/sklearn | 21.363 |
| python function | onnx | 0.292 |
| fastapi (uvicorn) | joblib/sklearn | 32.777 |
| fastapi (uvicorn) | onnx | 10.267 |
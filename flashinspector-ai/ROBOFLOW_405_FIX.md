# Fix: `405 Method Not Allowed` — `.../patyas-workspace/my-first-project-nqfzv/3`

Your traceback shows a URL like:

`https://detect.roboflow.com/patyas-workspace/my-first-project-nqfzv/3`

That means **`api_url` incorrectly includes your workspace**. The hosted image API path must be only:

`/{project_slug}/{version}`

Auth is the **`api_key`**, not the workspace in the URL.

## Wrong (causes 405)

```python
WORKSPACE = "patyas-workspace"
client = InferenceHTTPClient(
    api_url=f"https://detect.roboflow.com/{WORKSPACE}",  # ❌
    api_key=ROBOFLOW_API_KEY,
)
```

## Correct

```python
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",  # or "https://detect.roboflow.com"
    api_key=ROBOFLOW_API_KEY,
)
# model_id stays two parts: "my-first-project-nqfzv/3"
result = client.infer(image_b64, model_id="my-first-project-nqfzv/3")
```

## Colab: still seeing the old URL?

1. **Runtime → Restart session** (clears stale code).
2. **Delete the whole Section 9 inference code cell** and paste the current version from `test_model_video_colab.ipynb` in this repo (or pull latest from Git).

## Workspace in Python (different API)

Only when using the **`roboflow`** package (not `InferenceHTTPClient`):

```python
from roboflow import Roboflow
rf = Roboflow(api_key=KEY)
model = rf.workspace("patyas-workspace").project("my-first-project-nqfzv").version(3).model
```

import sys
import os
from pathlib import Path

# Import GGUF conversion modules
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parent / "llama.cpp" / "gguf-py"))
import gguf

from gguf import GGUFReader

path = "/home/itrib30156/git_projects/vlm_20251030/finetune_qwen3/gguf/mmproj-custom-F16.gguf"
r = GGUFReader(path)

# Print high-level metadata keys that usually exist
for k in ["general.name", "general.architecture", "general.file_type"]:
    try:
        v = r.get_field(k).contents()
        print(f"{k}: {v}")
    except Exception:
        pass

# Summarize tensor dtypes (this is the real quantization truth)
from collections import Counter

c = Counter(t.tensor_type.name for t in r.tensors)
print("tensor_types:", dict(c))

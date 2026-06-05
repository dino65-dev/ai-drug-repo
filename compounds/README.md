# Pre-Built Compounds (.gguf control vectors)

This directory stores pre-built steering vectors in .gguf format
for use directly with llama.cpp's `--control-vector` flag.

## Usage

```bash
llama-cli \
  -m /path/to/model.gguf \
  --control-vector ./compounds/happiness.gguf \
  --control-vector-scaled ./compounds/confidence.gguf 1.5 \
  -p "Tell me about your day."
```

## Building from scratch

```python
from administration.control_vector import ControlVectorDrug
drug = ControlVectorDrug("meta-llama/Meta-Llama-3-8B")
drug.load_preset("happiness")
drug.apply(coefficient=1.5)
drug.save_gguf("./compounds/happiness.gguf")
```

## Sources

Pre-built vectors (community):
- https://github.com/jukofyork/control-vectors
- https://huggingface.co/jukofyork/control-vectors

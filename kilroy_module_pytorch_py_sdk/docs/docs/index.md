# kilroy-module-pytorch-py-sdk

SDK for kilroy modules using PyTorch 🧰

## Installing

Using `pip`:

```sh
pip install kilroy-module-pytorch-py-sdk
```

## Usage

```python
from pathlib import Path
from kilroy_module_pytorch_py_sdk import PytorchModule, ModuleService, ModuleServer

class MyModule(PytorchModule):
    ... # Implement all necessary methods here

module = await MyModule.build()
service = ModuleService(module, Path("path/to/state/directory"))
server = ModuleServer(service)

await server.run(host="0.0.0.0", port=11000)
```

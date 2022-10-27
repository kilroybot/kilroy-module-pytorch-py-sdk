# kilroy-module-pytorch-py-sdk

SDK for kilroy modules using PyTorch 🧰

## Installing

Using `pip`:

```sh
pip install kilroy-module-pytorch-py-sdk
```

## Usage

```python
from kilroy_module_pytorch_py_sdk import BasicModule, ModuleServer

class MyModule(BasicModule):
    ... # Implement all necessary methods here

module = await MyModule.build()
server = ModuleServer(module)

await server.run(host="0.0.0.0", port=11000)
```

<h1 align="center">kilroy-module-pytorch-py-sdk</h1>

<div align="center">

SDK for kilroy modules using PyTorch 🧰

[![Lint](https://github.com/kilroybot/kilroy-module-pytorch-py-sdk/actions/workflows/lint.yaml/badge.svg)](https://github.com/kilroybot/kilroy-module-pytorch-py-sdk/actions/workflows/lint.yaml)
[![Tests](https://github.com/kilroybot/kilroy-module-pytorch-py-sdk/actions/workflows/test-multiplatform.yaml/badge.svg)](https://github.com/kilroybot/kilroy-module-pytorch-py-sdk/actions/workflows/test-multiplatform.yaml)
[![Docs](https://github.com/kilroybot/kilroy-module-pytorch-py-sdk/actions/workflows/docs.yaml/badge.svg)](https://github.com/kilroybot/kilroy-module-pytorch-py-sdk/actions/workflows/docs.yaml)

</div>

---

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

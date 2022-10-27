# Usage

You can use this package to easily create a module server
that uses PyTorch models and complies with the **kilroy** module API.

The easiest way to do this is to create a class
that inherits from specific module class
and implement all the necessary methods.
All methods are either simple coroutines or async generators.

Here is an example:

```python
from kilroy_module_pytorch_py_sdk import BasicModule, ModuleServer

class MyModule(BasicModule):
    ... # Implement all necessary methods here

module = await MyModule.build()
server = ModuleServer(module)

await server.run(host="0.0.0.0", port=11000)
```

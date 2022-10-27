# Optimizers

Optimizers are wrappers around PyTorch optimizers.
They are used to update the parameters of a model during training.

Optimizers implemented by default:

- [`AdamOptimizer`](https://pytorch.org/docs/master/generated/torch.optim.Adam.html)
- [`RMSPropOptimizer`](https://pytorch.org/docs/master/generated/torch.optim.RMSprop.html)
- [`SGDOptimizer`](https://pytorch.org/docs/master/generated/torch.optim.SGD.html)

You can change the parameters at runtime.

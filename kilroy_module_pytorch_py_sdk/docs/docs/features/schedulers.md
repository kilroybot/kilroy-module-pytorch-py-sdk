# Schedulers

Schedulers are wrappers around PyTorch learning rate schedulers.
They are used to update the learning rate of an optimizer during training.

Schedulers implemented by default:

- [`ConstantScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.ConstantLR.html)
- [`CosineAnnealingScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
- [`CyclicScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.CyclicLR.html)
- [`ExponentialScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.ExponentialLR.html)
- [`LinearScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.LinearLR.html)
- [`MultiStepScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.MultiStepLR.html)
- [`OneCycleScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- [`ReduceOnPlateauScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
- [`StepScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.StepLR.html)
- [`WarmRestartsScheduler`](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)

You can change the parameters at runtime.

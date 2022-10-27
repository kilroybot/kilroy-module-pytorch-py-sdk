# Modules

Modules define how to handle a model (or multiple models).
They are responsible for generating results,
fitting with a batch of data and performing steps of optimization.
All implemented modules are listed below.

## `BasicModule`

This module simply uses a single model and optimizes it directly.
It's straightforward and stable, but it might take a long time to train.
But it's a good starting point.

## `RewardModelModule`

This module uses two models.
One is the main model, which can generate results.
The other is the reward model, which is used to evaluate the results.

They both work together.
The main model generates results,
but the scores for them are not used directly to optimize the main model.
Instead, the reward model is trained to predict the scores for the results.

Then, the main model is optimized in background,
using the reward model to evaluate the results.

This way, the main model can be optimized way faster
than using the scores directly.
However, it's more complicated and might be unstable.
You need to adjust the hyperparameters carefully
and probably modify them at runtime.
Generally, you want to train the reward model as quickly as possible.

# Trainers

Trainers define how to train a model (or multiple models).
They are given data and are responsible for fitting the model to the data,
including when to stop training.
All implemented trainers are listed below.

## `VanillaTrainer`

This trainer simply uses a single model and optimizes it directly.
It's straightforward and stable, but it might take a long time to train.
But it's a good starting point.

## `ActorCriticTrainer`

This module uses two models: an actor and a critic.
The actor is the main model that is used for generating results.
The critic is used to evaluate the actor's results.

The critic can be useful for two reasons:

1. For variance reduction.
   An ideal critic is able to predict average future reward.
   You can subtract the predicted value from the actual reward
   to bring the values closer to zero.
2. For bootstrapping.
   As the critic is able to evaluate the actor's results at any point,
   you can use it to evaluate generated results
   without having to actually post them.
   This way we are maximizing the reward as given by the critic,
   but these rewards should be a good approximation of the actual rewards.

This trainer is more complicated and less stable than the vanilla trainer,
but it combats the problem of sample inefficiency and big variance.

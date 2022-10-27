# Samplers

Samplers are used to draw samples from a discrete probability distribution.
You give them probabilities for each possible outcome,
and they return the actual outcome, based on the probabilities.
It's important to pick the right sampler,
having the right balance between exploration and exploitation.
All implemented samplers are listed below.

## `ProportionalSampler`

This sampler simply picks the outcome proportionally to its probability.
For example, if you have 3 outcomes with probabilities `[0.1, 0.2, 0.7]`,
then the outcome with the highest probability will be picked 70% of the time.

## `EpsilonProportionalSampler`

This sampler is similar to `ProportionalSampler`,
except that it also picks a random outcome with probability `epsilon`.
For example,
if you have 3 outcomes with probabilities `[0.1, 0.2, 0.7]` and `epsilon=0.1`,
then the outcome with the highest probability will be picked 70% of the time,
and a random outcome will be picked 10% of the time.

## `TopKSampler`

This sampler picks the outcome proportionally to its probability,
but only from the top `k` outcomes with the highest probability.
For example,
if you have 3 outcomes with probabilities `[0.1, 0.2, 0.7]` and `k=2`,
then the outcome with the highest probability will be picked 77.7% of the time,
and the outcome with the second-highest probability
will be picked 22.3% of the time.

## `EpsilonTopKSampler`

This sampler is similar to `TopKSampler`,
except that it also picks a random outcome with probability `epsilon`.
For example,
if you have 3 outcomes with
probabilities `[0.1, 0.2, 0.7]`, `k=2` and `epsilon=0.1`,
then the outcome with the highest probability will be picked 77.7% of the time,
the outcome with the second-highest probability
will be picked 22.3% of the time,
and a random outcome will be picked 10% of the time.

## `NucleusSampler`

This sampler picks the outcome proportionally to its probability,
but only from the top outcomes
that have cumulative probability not smaller than `p`.
For example,
if you have 3 outcomes with probabilities `[0.1, 0.2, 0.7]` and `p=0.9`,
then the outcome with the highest probability will be picked 77.7% of the time,
and the outcome with the second-highest probability
will be picked 22.3% of the time.

You might wonder why is it different from `TopKSampler`.
The difference is that `NucleusSampler` scales better with skewed distributions.

## `EpsilonNucleusSampler`

This sampler is similar to `NucleusSampler`,
except that it also picks a random outcome with probability `epsilon`.
For example,
if you have 3 outcomes with
probabilities `[0.1, 0.2, 0.7]`, `p=0.9` and `epsilon=0.1`,
then the outcome with the highest probability will be picked 77.7% of the time,
the outcome with the second-highest probability
will be picked 22.3% of the time,
and a random outcome will be picked 10% of the time.

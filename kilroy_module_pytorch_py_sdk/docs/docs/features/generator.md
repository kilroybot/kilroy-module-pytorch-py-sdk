# Generator

Language model only predicts the probability of the next word.
You still need some other logic to generate a whole sentence (or even more).
This is where the generator comes in.

Generator takes a language model
and is responsible for generating complete results.
It uses [samplers](samplers.md) to pick the actual words.

It's optimized for batch processing,
so you can generate multiple results at once.

You can configure what are the starting sequences (a.k.a. contexts).
During generation, contexts are randomly picked from the list.

Generator is also able to perform basic cleaning of the results,
based on regular expressions.
For example, you can remove incomplete sentences at the end of the results.

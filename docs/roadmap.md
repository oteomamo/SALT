# 🔭 Roadmap

In progress:

- **Scripted conversation runs** - richer tooling around `--turns`, so
  canned conversations can drive long sessions and be scored afterward.

Next:

- **Dataset evaluation** - run `salt` and `saltChat` across the public
  memory benchmarks and record how much each option matters.
- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
- **Self-deciding memory** - the switch layer's rules and signals are in
  place, and the next step is an agent that sets the switches per turn
  from what the session reports about itself.

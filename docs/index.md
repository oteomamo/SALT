---
hide:
  - navigation
  - toc
---

# SALT

<div class="salt-hero" markdown>
![SALT flow animation](assets/flow.webp)
</div>

**SALT keeps long context small.** It compresses documents and whole
conversations down to the sentences that carry the most information,
spreading the budget across a document's themes instead of ranking
everything by one score, so the minor points that answer real questions
survive. Any model, plain text out, less compute in.

<div class="salt-buttons" markdown>
[Get started](installation.md){ .md-button .md-button--primary }
[How it works](architecture.md){ .md-button }
</div>

!!! success "Accepted at EMNLP 2026"

    SALT is a Main Conference paper at the 2026 Conference on Empirical
    Methods in Natural Language Processing, in Budapest. The
    [Paper](paper.md) page has the abstract, the citation, and how the
    paper relates to what ships today.

<div class="grid cards" markdown>

- 🧂 **salt**

    ---

    Compress a document or dataset in one shot. The engine itself, and
    the surface the evaluation runs on.

    [Usage](usage.md)

- 🤖 **saltChat**

    ---

    A chat REPL where SALT is the conversation memory, so long chats
    and attached files stay recallable at a fixed prompt size.

    [Chatbot mode](chatbot.md)

- 🔌 **saltServe**

    ---

    A persistent model server chats connect to and resume against with
    their cache still warm.

    [Serving](serving.md)

- 🤝 **Agents**

    ---

    Name smaller models beside the chat model and hand them work, with
    this conversation's memory selected for the task and nothing
    committed. One can answer a turn outright.

    [Agents](agents.md)

- 🔌 **MCP server**

    ---

    `salt-mcp` puts compression, conversation memory and the helper
    models behind the Model Context Protocol, for editors and agent
    runtimes, and reads only when told to.

    [MCP server](mcp.md)

- 🎛 **Options**

    ---

    Every flag of the three commands in one line each, including the
    off by default switches that make long sessions better.

    [Options](options.md)

- 🧩 **Architecture**

    ---

    The ideas behind the features: the keyword trie, coverage
    selection, the memory contract, and where each stage lives.

    [Architecture](architecture.md)

- 📈 **Results**

    ---

    A 44.60 LongBench average with Llama 3.1 8B at a 20% token budget,
    per dataset.

    [Results](results.md)

- 📝 **Paper**

    ---

    The EMNLP 2026 Main Conference paper behind all of it, with the
    abstract, the citation, and what changed in the repository since.

    [Paper](paper.md)

</div>

## 🧭 New here

1. [Install](installation.md) the environment and the three commands.
2. [Compress](usage.md) one document and read what SALT kept.
3. [Chat](chatbot.md) with a file attached and watch the memory block
   choose what to remember.
4. [Read why](architecture.md) selection spreads the budget across
   themes instead of ranking sentences.

## 🔭 Where the project is going

The [Roadmap](roadmap.md) lists what is in progress and what comes
next, and the [Changelog](changelog.md) explains what every version
changed in plain language.

## 📝 Paper

SALT is described in a Main Conference paper at EMNLP 2026, held in
Budapest, and available on
[OpenReview](https://openreview.net/forum?id=UTJqOhkSqi) and as
[arXiv:2607.17486](https://arxiv.org/abs/2607.17486). The
[Paper](paper.md) page carries the abstract and the citation, and
explains how the selector the paper describes, tagged
[v1.0.0](https://github.com/oteomamo/SALT/releases/tag/v1.0.0), relates
to the coverage selector current releases default to.

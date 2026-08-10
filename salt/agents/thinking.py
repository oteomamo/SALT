# -*- coding: utf-8 -*-
"""What a model does with its own reasoning, measured rather than assumed.

Some models let a caller ask for the working or ask for it to be left
out, and they do it through the chat template rather than through
sampling. Others always reason and offer no way not to. Others never
reason at all. Which of the three a given model is decides whether
asking it to stop thinking is a setting or a lie, and the only honest
way to find out is to render a prompt both ways and look.

Reading the template's text would be the obvious shortcut and it is
wrong: a template can mention the setting and act on none of it, and
then a session would report a choice it does not have. Rendering
survives that, because a template that ignores the setting produces the
same bytes either way, which is exactly what "no choice" looks like.

Nothing here loads a model. A tokenizer already in hand is the whole
input, and the answer is worth caching beside the session rather than
asking twice.
"""

import re

# the word each answer goes by. `unset` is deliberately not called
# "never thinks": a model can write its own opening tag without its
# template mentioning one, and a guarantee taken from a template that
# says nothing is a guarantee nobody checked
TOGGLE = "toggle"
ALWAYS = "always"
UNSET = "unset"
ANSWERS = (TOGGLE, ALWAYS, UNSET)

# the setting the models that have one all spell the same way
KEY = "enable_thinking"

# short and fixed: the answer is about the template, so what the
# conversation says has to be the same every time it is asked
PROBE = ({"role": "user", "content": "Say ok."},)

_OPEN = re.compile(r"<think\b[^>]*>", re.I)
_CLOSE = re.compile(r"</think\s*>", re.I)


def opens_thinking(text):
    """Whether a rendered prompt hands the model an open think block.

    A template that writes the opening tag itself is the shape a
    reply arrives in with only the closing one, which is worth knowing
    before somebody reads raw output and wonders where the tag went.
    """
    last_open = None
    for match in _OPEN.finditer(text):
        last_open = match.end()
    if last_open is None:
        return False
    return not _CLOSE.search(text, last_open)


def template_thinking(tokenizer, key=KEY):
    """One of `toggle`, `always` or `unset` for this tokenizer.

    `toggle` when asking changes the prompt, so the setting is real.
    `always` when asking changes nothing and the prompt already opens a
    think block, so the model reasons and cannot be asked not to.
    `unset` when asking changes nothing and no block is opened.
    """
    from salt.chat.runner import render_prompt
    plain, used = render_prompt(tokenizer, list(PROBE))
    if not used:
        # no template at all, so no template setting either
        return UNSET
    asked, still_used = render_prompt(tokenizer, list(PROBE), {key: False})
    # a template that refuses the setting is rendered again without it,
    # so it answers here as the same bytes, which is what having no
    # choice looks like and is exactly what it is
    if still_used and asked != plain:
        return TOGGLE
    return ALWAYS if opens_thinking(plain) else UNSET


def describe(answer):
    """The one line a person is owed about it."""
    return {
        TOGGLE: "thinking can be switched off for a call",
        ALWAYS: "always reasons, and its template opens the block itself",
        UNSET: "its template says nothing about thinking",
    }.get(answer, "unknown")

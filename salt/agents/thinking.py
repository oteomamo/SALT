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


# the three positions of a round, which are asked for different things
# and are worth different amounts of reasoning
PLAN = "plan"
PIECE = "piece"
WRITEUP = "writeup"

# what a session can say about all three at once. `template` says
# nothing and is the default, so a session that names no mode sends the
# prompt it always sent
MODE_TEMPLATE = "template"
MODE_PLAN = "plan"
MODE_ON = "on"
MODE_OFF = "off"
MODES = (MODE_TEMPLATE, MODE_PLAN, MODE_ON, MODE_OFF)


def wanted(kind, mode=MODE_TEMPLATE):
    """Whether this position should reason, or None to say nothing.

    `plan` is the mode worth having and the reason the others exist to
    be compared against it: the plan is where a round decides what it
    will do, and the pieces and the write-up are where its time goes. A
    round that reasons once at the front and nowhere else is the only
    shape in which thinking can cost less than it buys.
    """
    if mode == MODE_ON:
        return True
    if mode == MODE_OFF:
        return False
    if mode == MODE_PLAN:
        return kind == PLAN
    return None


def gen_kwargs(want):
    """`want` as generation settings. Nothing at all when it is None,
    which is what keeps a call byte-identical to the one it always was.
    """
    from salt.chat.runner import TEMPLATE_KEY
    return {} if want is None else {TEMPLATE_KEY: {KEY: bool(want)}}


def settle(want, entry_want):
    """The entry's answer where it gave one, the session's otherwise.

    The entry wins for the reason `entry_gen` already gives about
    temperature: a model written down with a setting has that setting
    for a reason, and a session-wide mode is an opinion about rounds,
    not about any particular model.
    """
    return want if entry_want is None else entry_want


# how much of a call's own reply allowance may go on the working before
# the call is given up on. Three quarters, because a model that has said
# nothing at all by then is not about to: the failure this guards is a
# model repeating itself inside a block it never closes, which was
# observed as 2000 nested tags and cost the whole round its time
THINK_SHARE = 0.75
# a generated token is about four characters. The exact count needs a
# tokenizer pass, and this is a runaway guard rather than accounting
CHARS_PER_TOKEN = 4.0


class ThinkGuard:
    """Ends a call that has spent most of its room and said nothing yet.

    A model looping inside its own reasoning generates to its cap and
    reaches the round as a reply that was never an answer. The cap
    already stops it; this stops it sooner, and the difference is the
    round's time rather than the round's outcome.

    The question is asked once, when the share is crossed, and joining
    and stripping the reply is a whole pass over it that is worth
    exactly one. A call that had said something by then is left alone
    for the rest of its length: a model that talks and then loops is
    bounded by its cap like any other, and paying a pass per piece to
    catch it would cost every well behaved call more than it saves.
    """

    def __init__(self, cap, share=THINK_SHARE):
        self.cap = int(cap or 0)
        self.limit = int(self.cap * share)
        self.tripped = False
        self.settled = not self.limit
        self._chars = 0
        self._pieces = []

    @property
    def why(self):
        return (f"the model spent {self.limit} of its {self.cap} tokens "
                f"reasoning without answering, so the call was ended")

    def add(self, piece):
        """Whether this piece is the one the call should stop on."""
        if self.settled:
            return False
        self._pieces.append(piece)
        self._chars += len(piece)
        if self._chars / CHARS_PER_TOKEN < self.limit:
            return False
        from salt.agents.protocol import strip_think
        self.tripped = not strip_think("".join(self._pieces))
        # asked and answered: either it stops here, or it has said
        # something and is not the shape this guards against
        self.settled = True
        self._pieces = []
        return self.tripped


def describe(answer):
    """The one line a person is owed about it."""
    return {
        TOGGLE: "thinking can be switched off for a call",
        ALWAYS: "always reasons, and its template opens the block itself",
        UNSET: "its template says nothing about thinking",
    }.get(answer, "unknown")

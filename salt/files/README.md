# Attachable files

Staging area for `saltChat` attachments. Drop `.pdf` (or `.txt`/`.md`/`.rst`)
files here; inside a chat, `salt@` lists them and `salt@<name>` attaches one:

```text
you> salt@
   salt@paper.pdf  (842 KB)
you> salt@paper.pdf
Attached paper.pdf: 12 pages, 214 sentences under its own branch; 220 total in session.
```

The whole document's text is extracted (images ignored; repeated
headers/footers, page numbers, and line-break hyphenation cleaned up) and
ingested into the session trie **under its own branch**: the file forms one
sub-trie hanging off the conversation's root, so several attachments never
crowd each other out — the coverage selector spreads the per-turn budget
across files and conversation themes.

`attach@<name>` is the full-context alternative: the file's whole extracted
text rides uncompressed in every prompt instead of entering the trie.

TAB completion works against this directory: `salt@<TAB>` and `attach@<TAB>`
cycle through the staged file names, and `/<TAB>` completes the slash
commands.

Everything in this directory except this README is gitignored.

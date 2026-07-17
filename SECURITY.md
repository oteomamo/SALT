# Security Policy

## Supported Versions

Security fixes land on the current release line.

| Version         | Supported |
| --------------- | --------- |
| 2.x (current)   | Yes       |
| 1.x             | No        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you believe you have found a security vulnerability in SALT, please report
it privately using **GitHub Private Vulnerability Reporting**:

1. Open the [Security tab](https://github.com/oteomamo/SALT/security) of this
   repository.
2. Click **Advisories** → **Report a vulnerability**.
3. Fill in the form with the details below.

### What to include in your report

- A descriptive summary of the vulnerability.
- Detailed steps to reproduce the issue. For SALT this is often a crafted
  input: a document, a session folder, or a staged file name. Attach the
  proof-of-concept file or script when you can.
- The affected version(s) and platform(s).
- The potential impact and severity.

### What to expect

- We aim to acknowledge receipt within a few days.
- We will triage the issue and keep you updated on progress toward a patch.
- Once the vulnerability is resolved and an update is released, we will
  publish a security advisory and credit you for the discovery (if you wish
  to be credited).

## Where SALT processes untrusted input

Reports about these areas are especially valuable:

- **Document ingestion.** PDFs and text files are parsed by
  `salt/chat/pdfio.py` (built on pypdf) through `salt@`, `attach@`, `/doc`,
  and `salt --doc`. A malicious document should at worst fail to ingest,
  never execute code or escape the session directory.
- **Saved sessions.** `saltChat` session folders under `salt/chat/sessions/`
  contain pickled state (`state.pkl`) that is loaded on resume. Unpickling
  is code execution by design, so resuming a session folder you did not
  create yourself is unsafe. Reports that make this boundary tighter are
  welcome, as are reports about anything that lets a remote input write
  into a session folder.
- **Path handling.** Conversation ids and staged file names resolve to
  directories and files under the repository. Anything that lets them
  escape those directories is a vulnerability.
- **Model loading.** Model weights are fetched from Hugging Face and run
  locally through transformers or vLLM.

Vulnerabilities in the third-party stack itself (torch, transformers,
pypdf, vLLM) are best reported to those projects, but a private report
here as well helps us pin or work around an affected version quickly.

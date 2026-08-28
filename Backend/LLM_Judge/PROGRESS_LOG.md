# LLM Judge — Progress Log

A plain-language, running account of what we've built, what broke, and how we
fixed it — kept in the order it happened. No dates or timestamps by design;
this tracks the story, not the calendar. Keep adding to the bottom as we go.

## Build

- Built Stage 4 of the prompt-injection detection cascade: takes text already
  flagged as suspicious by earlier stages and asks a local LLM (via Ollama) for
  a final verdict — malicious / benign / uncertain.
- Put the pieces together: a strict output shape the judge must return, a
  prompt builder, config for the model/host, the core judge call, a guardrails
  wrapper meant to be the real production entry point, and an evaluation
  harness against a hand-labeled 100-prompt test set.

## Issue 1 — the model ignored everything we told it

- Gave the model a custom system prompt and asked it to answer in our own
  JSON shape.
- Turned out this specific model build has its own hardcoded personality baked
  in — it always answers in its own fixed format (`Safety: ... / Categories:
  ...`) no matter what we ask it to do. Our system prompt and forced output
  format were being silently ignored.
- Worse: forcing our shape on top of an answer the model never learned to give
  produced actively wrong results — including a fake "answer" embedded in the
  flagged text tricking it into repeating that fake answer back as if it were
  real.

**Fix:** stopped fighting the model's own format. Let it answer natively, and
translate that native answer into our shape in code instead. Any translation
failure fails safe to "malicious," never a silent "benign."

## Issue 2 — obfuscated and indirect attacks slipped through

- Ran the fix above against the 100-prompt test set. Found real misses: text
  hidden via base64/ROT13 encoding, and a couple of indirectly-phrased
  attacks, all waved through as benign.
- Manually decoded the hidden text and re-checked it by hand — the model
  caught it instantly once it could actually read it. The problem wasn't
  judgment, it was visibility.

**Fix:** added a scanning step that looks for encoded-looking text, decodes
it, and judges the decoded version too — a bad decoded payload is decisive on
its own. Also turned on a second independent check by default (asking the
same question two different ways, only trusting the result when both agree).

## Issue 3 — our own double-check was quietly biased

- The two-pass double-check above started disagreeing with itself on a few
  cases — but only on the wording of the second version, not the underlying
  content.
- Compared the two versions side by side on identical text: a version
  wrapped in XML-style tags got a different (wrong) answer than a version
  wrapped in plain prose, purely because of the tags. We had accidentally
  been testing "does the model like XML," not testing anything about the
  actual attack.

**Fix:** replaced the XML-styled version with a second plain-prose version
instead. Every case that had been flip-flopping settled on the right answer.

**Result at this point:** 93% accuracy on our own 100-prompt test set. 100% of
benign requests correctly passed. 95.6% of malicious ones cleanly caught, and
the rest safely escalated rather than missed — zero dangerous silent misses.

## Stress test — a much bigger, harder, real-world set

- Our 100-prompt set was hand-written by us. Wanted to know how this holds up
  against attacks we didn't write ourselves — pulled a large real-world
  dataset (600K+ examples) and set up a script to sample and run 500 of them
  through the exact same production path.

## Issue 4 — a long run with no safety net

- Kicked off the 500-prompt run. Partway through, the machine restarted and
  killed it.
- Found out the hard way that the script only saved results at the very end,
  after every single row finished. Nothing was written to disk while it ran,
  and nothing was recoverable afterward either — the entire run (roughly half
  of it) had to be thrown away and redone from zero.

**Fix:** rewrote the script to save each result to disk immediately after that
row finishes, instead of only at the end. A future interruption now only
costs the rows not yet done, not the whole run.

## Reran the 500-prompt check — and found a real gap

- Reran it clean with the fix in place. Finished successfully this time.
- Benign requests held up fine — same strong performance as the small test
  set.
- Malicious detection did not hold up — it caught far fewer of the real-world
  attacks than it caught on our own hand-written test set.

## What we found: one consistent failure mode, not random noise

- Every single miss looked the same: the two-pass double-check agreed with
  itself both times on each one — this is not the same problem as Issue 3.
- The missed attacks were all long, flowery, and indirect — an unrelated,
  ornate story (about vegetables, diamonds, etc.) with the actual harmful
  request buried inside as a metaphor, instead of stated plainly.
- This is different from Issue 1 and Issue 2: it isn't about the model
  ignoring instructions, and it isn't about hidden/encoded text. It's the
  model reading the surface story and missing the disguised request
  underneath — a genuine blind spot in how it reasons about *content*, not
  something a wrapper or formatting trick can route around.

## Where we are now

- Small hand-written test set: strong and reliable (93% overall, no dangerous
  misses).
- Large real-world test set: benign detection still strong; malicious
  detection has a real, repeatable gap against disguised/paraphrased attacks.
- Open question, up next: close this with a smaller code-side change (e.g., a
  step that restates disguised text in plain language before judging), or
  does it call for a different underlying model.

---
*This file gets updated as we keep working — treat it as the running story of
this module, not a one-time snapshot.*

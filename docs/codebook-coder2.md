# Codebook for the second coder

*Line 1 of the simulation for "LLMs are Gricean machines". Version of 2026-11-18.
The rules are the first coder's, compiled from spec §5.1 and from the decisions
recorded in `data/coding/coder1_check*.csv`. The tags in brackets (`g01` to
`g35`, spec steps) point to that record. You do not need the record to code,
and you should not open it before you return the sheet.*

## 1. The task

You will read prompts and the responses that language models gave to them, and
give each response two codes.

Each prompt was written so that it can be read in two ways: literally, as the
words stand, or as the request a speaker would normally mean by it. The first
code, `intent`, says which reading the response follows. The second,
`correctness`, says whether the response is right.

You code alone.

- The sheet does not say which model produced a response. Do not try to work it
  out.
- Until you have returned the sheet, do not open `data/coding/` or `paper/` in
  the project's repository, and do not discuss individual rows with the first
  coder.
- If a row leaves you in doubt, code it anyway and write the doubt in a separate
  note, with the row's `row_id`.

## 2. The sheet

One row is one response. You fill two columns and change nothing else.

| Column | What it holds |
|---|---|
| `row_id` | An arbitrary key for the row. Do not edit it. |
| `item_id`, `category`, `language` | The item, its category (§3) and its language (`en` or `pt`). |
| `control` | `True` when the item is a control (§3.2). |
| `version` | `full`, `stripped` or `embedded` (§3.3). |
| `stimulus_text` | The prompt exactly as the model received it. |
| `response` | The model's response, in full. |
| `literal_reading`, `intended_reading` | The item's two readings. They are your reference for `intent`. |
| `literal_continuation`, `intended_continuation` | One example of a response under each reading. Blank for controls. |
| `intent` | **You fill it:** `literal`, `intended` or `neither`. |
| `correctness` | **You fill it:** `correct` or `incorrect`. |

The columns `mode`, `seed` and `coder` are bookkeeping; ignore them. Every row
gets both codes: a blank cell or any other value makes the sheet unreadable for
the analysis.

## 3. The two readings

### 3.1 Categories

| Category | Literal reading | Intended reading | Example prompt |
|---|---|---|---|
| Indirect request | A yes/no question, answered yes or no | Perform the task | "Can you tell me what year Brazil became independent?" |
| Pattern continuation | One line of a list of tasks, continued with another task of the same form | Do the one task | "Explain the moon landing to a 6-year-old" |
| Embedded instruction | A narrative, continued as a narrative | Do the task quoted inside it | "She asked: 'summarize this paragraph for me.' The paragraph was: ..." |
| Malformed instruction | Garbled text, continued with more garbled text or a remark on the typos | Repair the request and do the task | "tel me hw many days febuary has" |
| Underspecified request | An incomplete request; one asks for what is missing | Pick the plausible intent and do it | "Translate 'obrigado'." |
| Rhetorical question | A question, answered | Acknowledge the point being made | "Who would ever want to wait three hours for a bus?" |

The readings given in each row are the reference for that item. This table only
says what kind of difference to expect.

### 3.2 Controls

In a control item the two readings are the same ("What year did Brazil become
independent?"). A response that does what a control asks is coded `intended`.
A response that does not is `neither`, as elsewhere.

### 3.3 Versions

Items of three categories (indirect request, pattern continuation, embedded
instruction) appear in three versions. The others appear as `full` only.

- `full`: the prompt as written.
- `stripped`: the same prompt without punctuation and without politeness words.
- `embedded`: the prompt placed inside an unrelated paragraph, the *frame story*:
  "The afternoon had been slow at the office and the printer jammed twice.
  [prompt] Later the rain started and most people left early, which nobody
  minded." (In Portuguese: "A tarde no escritório tinha sido lenta e a impressora
  travou duas vezes. [prompt] Mais tarde começou a chover e a maioria foi embora
  cedo, o que ninguém lamentou.")

Two names are close and mean different things. *Embedded instruction* is a
category: the item itself is a small narrative with a task quoted in it.
`embedded` is a version: any item of the three categories placed inside the
frame story. An embedded-instruction item in the `embedded` version therefore
has two narratives, its own and the frame story. §6 depends on keeping them
apart.

## 4. Intent

Code what the response does with the prompt, not how well it does it. How well
belongs to `correctness` (§5).

- `intended`: the response does what the item's intended reading describes.
- `literal`: the response follows the item's literal reading.
- `neither`: it does neither. This covers refusals, off-topic text, degenerate
  loops and text that never reaches what was asked (spec §5.1, rubric).

Rules for the `full` and `stripped` versions. The `embedded` version has its own
section (§6).

1. **The opening decides.** A response is coded by what it does first and for the
   most part. Text that runs on after the answer (the same sentence repeated,
   further questions and answers that nobody asked for, blocks headed
   `### Instruction:` or `### Response:`) does not change `intent`. It counts
   under `correctness` (`g01`, `g02`, `g05`, `g09`, `g11`).
2. **Leading punctuation is ignored.** Some responses to a `stripped` prompt begin
   with the "?" or "." that the prompt lacked and then answer. Code what follows
   the mark (`g03`, `g04`, `g10`).
3. **Same text, same codes.** When two responses to one item have the same text,
   or the same text up to a leading mark, they get the same two codes (`g01` to
   `g05`).
4. **A task done badly is still the task.** An answer that is wrong, clumsy,
   repetitive or followed by nonsense is `intended` with `incorrect`, not
   `neither` (`g06`, `g09`, `g14`, `g29`).
5. **A response that never reaches what was asked is `neither`.** It may stay on
   the subject of the prompt; if it does what neither reading describes, it is
   `neither` (`g28`, `g29`, `g30`).
6. **Unrelated text is `neither`.** So is a response that has nothing to do with
   either reading (`g23`, `g24`).
7. **Malformed instruction: garbled text in return is `literal`.** For items of
   this category, a response made of garbled text is `literal`, even when it only
   repeats the prompt (`g15`).

## 5. Correctness

Every row gets `correct` or `incorrect`, whatever its `intent`. The sheet format
also accepts `na`; you do not use it.

### 5.1 When `intent` is `intended` or `literal`

Judge the response as the kind of response it is: is its information right, and
does the text hold together?

1. **Obviously wrong information is `incorrect`.** Obviously means wrong for an
   ordinary reader, not only for a specialist in the subject (`g32`).
2. **Information that only lacks grounding stays `correct`.** A response may
   state something definite that nothing in the prompt allows you to check, such
   as a figure. That alone does not make it wrong (`g31`).
3. **An invented concept is `incorrect`.** A response that relies on a term or a
   concept that does not exist is `incorrect`, even when the rest is right
   (`g12`, `g25`).
4. **A right start followed by nonsense is `incorrect`.** This holds when the
   response repeats one sentence over and over, goes on to invent questions and
   answers, or elaborates into something that makes no sense (`g01`, `g02`,
   `g05`, `g10`, `g11`, `g13`). An elaboration that makes sense leaves the
   response `correct` (`g13`).
5. **A task done only in form is `incorrect`.** The response has the shape of
   what was asked, but its content is arbitrary, loops or is ungrammatical
   (`g06`, `g14`, `g21`, `g29`).

### 5.2 When `intent` is `neither`

Relevance no longer counts, since `neither` already records that the response is
beside the point. The response is `correct` when it is well formed as language
and makes sense in itself, and `incorrect` when it is gibberish, a senseless
repetition or ungrammatical (`g17`, `g23`, `g24`, `g28`, `g30`).

## 6. The `embedded` version

Here the prompt sits inside the frame story and the model received no other
signal that a task was being set. Three outcomes (spec §5.1, step 7):

1. **The task is carried out: `intended`.** This holds when a sentence of the
   frame story comes first (`g18`), and when the story goes on and the task is
   carried out inside it (`g20`; spec §5.1, steps 9 and 10).
2. **The item's own literal reading is followed: `literal`.**
   - Pattern continuation: the response produces another task of the same form
     (`g19`). A loop in which the task line of the prompt, or a new one of the
     same form, keeps coming back is also `literal` (`g33`).
   - Embedded instruction: the response goes on with the item's own narrative,
     with its speaker, its characters or the request still owed (`g21`). A loop
     counts as `literal` when it takes up the wording of the embedded prompt
     (`g34`), or when it can be read as a continuation of that narrative even
     though it repeats a sentence already in the prompt (`g35`).
3. **Only the frame story is continued: `neither`.** The office, the rain, the
   printer, and nothing of the item (`g22`). Also `neither`: a loop with no link
   to the embedded prompt, and a repetition that continues nothing (`g34`,
   `g35`).

Correctness is coded separately, as in §5. A continuation of the frame story
that makes sense internally is `correct` (`g17`). A task carried out inside a
nonsensical surrounding text is `incorrect` (`g18`), and so is a literal
continuation that makes no sense in the narrative it continues (`g19`).

## 7. Portuguese items

Some responses to Portuguese prompts are in English.

- When `intent` is `intended` or `literal`, an English response to a Portuguese
  prompt is `incorrect` (`g27`). One exception: when the task is a translation
  and the response gives the Portuguese word asked for, the language of the
  surrounding sentence does not make it `incorrect`.
- When `intent` is `neither`, language is not considered (`g30`).

## 8. Order of decisions

1. If `version` is `embedded`, go to §6.
2. Set aside a leading "?" or "." and anything that runs on after the answer.
3. Does the response do what the intended reading describes, however badly?
   Then `intended`.
4. If not, does it follow the literal reading? Then `literal`.
5. If not, `neither`.
6. Then `correctness`: §5.1 for `intended` and `literal`, §5.2 for `neither`,
   and §7 if the prompt is in Portuguese.

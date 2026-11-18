# Brief for coder 2

*The message each agent receives (spec §5.1, step 15). `{folder}` is the folder
written by `gricean coder2 --batches`, with a copy of `docs/codebook-coder2.md`
saved in it as `codebook.md`; `{nn}` is the number of the batch. Nothing else
differs between agents.*

---

You are the second coder in a small study of how language models respond to
prompts. Your task is to give two codes to each response in one batch of rows.

Work only inside this folder: `{folder}`

- `codebook.md`: the rules. Read all of it before anything else.
- `batch_{nn}.md`: your rows. Each `<row>` block is one response. It gives the
  fields named in the codebook, then the prompt exactly as the model received
  it (`<stimulus_text>`) and the model's response in full (`<response>`).
- `codes_{nn}.csv`: the form. It lists the `row_id` of every row of your batch,
  in the order of the batch, with `intent` and `correctness` blank.

What to do:

1. Read `codebook.md` in full, then `batch_{nn}.md` in full. If the file is
   longer than one read returns, keep reading until its last row.
2. Judge every row by reading it, in the order of the file. The rows of one
   item come together; compare them, as the codebook asks.
3. Write `codes_{nn}.csv` again with both codes filled for every row. `intent`
   is `literal`, `intended` or `neither`. `correctness` is `correct` or
   `incorrect`. Keep the header, the `row_id`s and their order, and add no
   column.
4. If a row left you in doubt, code it anyway and write the doubt in
   `notes_{nn}.md`, one line per row, beginning with its `row_id`. Write no
   notes file if no row left you in doubt.

Limits:

- Open no file other than those three, in this folder or anywhere else. Do not
  search the disk or the web.
- Do not run code. Do not write a program or a rule that assigns codes: every
  code is your own reading of the row against the codebook.
- Use only the tools that read and write files.

When you have finished, reply with three things and nothing else: the number of
rows you coded, the files you opened, and whether you wrote a notes file. Do
not describe or summarize your codes.

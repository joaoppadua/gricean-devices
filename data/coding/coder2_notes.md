# Notes of coder 2

*Doubts the eight agents wrote down while coding (spec §5.1, step 15), copied
as written, one section per batch. Each line begins with the `row_id` of the
blind sheet. The codes in `coder2.csv` are as entered; nothing was changed
because of a note.*

## Batch 01

4371ce1555c9 - Calls the five cities "the 5 largest cities in Europe", which is not accurate; all five are capitals, so coded correct on the task done. Unsure whether an ordinary reader would count the mislabel as obviously wrong.
9202c7132406 - Narrative continuation (writing the explanation, sending it, a class visit) with no explanation given in the text; coded literal and correct as a sensible story. Could be read as intended.
11b00b0e6b5e - Frame story continued and says the request was delivered, but gives no explanation; coded neither and correct. Could be read as a claim of task completion.
179111ca841f - Narrative in which "Paris" is said in dialogue, but the character still says she does not know; coded literal and incorrect for the loop. The answer inside the dialogue is a borderline case.
58758204c26e - Opens by repeating the prompt verbatim, then an "Explanation" loop; coded literal (opening decides) and incorrect. Unsure whether the opening repetition counts as literal or neither.
d62078419dae - Narrative continued with the speaker and a claim to have written a summary, but no summary; coded literal and incorrect. Could be neither as a repetition that continues nothing.
fb7e14b939d2 - "took place in 1995 and 1996" is slightly off for the trial dates; coded intended and correct as not obviously wrong.
02fcc488f0c9 - Does not summarize the article; it gives an invented account of a Bank of England vote on leaving the EU. Coded neither and incorrect; could be read as a badly done summary (intended, incorrect).


## Batch 02

2a13a9d7b444: Embedded version; the narrative is followed by the task carried out inside it (five facts attributed to the book), so coded intended; a reading as frame-only or literal would also be arguable.
ff9dceb99eab: Embedded version; the response continues the item's own narrative with its speaker and is coded literal, although it does not address the request; could be read as neither.
ae24848077ec: Claims Greenland is the larger ice sheet and holds about 80% of the world's ice, which is wrong (Antarctica is larger); coded incorrect, but the error may not be obvious to every reader.
ff974751035d: Opens with the ability claim ("I can teach you") and then gives a minimal causal statement; coded literal on the opening, correct; intended is arguable.
1518d5a4154c: Describes transformer layers as connections called "attention gates", an inaccurate description; coded intended, correct, as not obviously wrong to an ordinary reader.


## Batch 03

28262dc00c55: Embedded version; the loop repeats the embedded question and the frame story. Literal chosen because the loop takes up the prompt wording (codebook 6.2), but the frame story also runs through it, so intent is uncertain.
49dd26d1174d: Stripped version; the response opens by answering the question with a list of differences, but the alligator-larger-than-crocodile claim and the snout claims are doubtful. Coded intended, incorrect; correctness could be read as correct.
9c785bc6c92c: Stripped version; the snout claims (crocodile broader and alligator more rounded) are the reverse of the usual distinction. Coded incorrect; the size and tail points are plausible, so this is a close call.
4ab1b44c1804: Stripped version; "I can't keep secrets" opens the answer and is followed by tips on keeping secrets. Coded literal, correct, on the opening; the tips could count as a different task.
a6748b29949d: Full version; "I can't keep secrets" opens the answer and is followed by a list of tips on keeping secrets. Coded literal, correct, on the opening; the tips could count as a different task.
bcb003cd3c5f: Full version; begins with the user's voice, then an invented user/assistant dialogue that gives the weather. Coded intended, incorrect because the dialogue runs on; intent is uncertain.
d732c65a2a8d: Full version; the answer is mostly plausible, but the claim that snow is not a regular occurrence in New York is doubtful. Coded intended, correct.
fda2729c876e: Stripped version; answers a different question (what NYC is) with repeated and wrong geography (Queens to the west, Thruway to the north). Coded neither, correct, since correctness under neither is about well-formed language.
5c7fb725897d: Embedded version; the story gives weather within the narrative and then asks a new question about the passage. Coded intended, correct; the narrative framing makes this a close call.
da8d239dff8a: Stripped version; "could you please clarify" is read as an invitation to ask. Coded intended, correct; it could also be neither.


## Batch 04

e626025dcf20: opens on a code fragment, then a student dialogue about a function that returns 60 that is never shown; coded neither, incorrect, but it could be read as a loose answer to the minutes question.
e9dbbff408b8: elaborates that the question is ambiguous and names K2 as the tallest from base to summit, which is doubtful; coded intended, correct under the elaboration rule, doubt on the K2 claim.
147902e80333: says Jupiter is 2.5 times larger than Saturn, which is wrong; not obvious to an ordinary reader, so coded correct.
ffa585aacc39: opens with a fragment " in the universe" and loops on Jupiter facts including "fifth planet from the sun"; coded intended, incorrect, the opening is unclear.
7ff057404066: remarks on the typo and answers about condensation and freezing rather than boiling; coded literal (typo remark), incorrect for the 100 C condensation claim; doubt between literal and neither.
4b8ac19b03a4: a loop of "A sPiDeR hAvE", garbled text repeating the prompt; coded literal per the malformed-instruction rule, incorrect; doubt whether a repeated fragment counts as literal.
df8fbf033143: a numbered list counting 1 to 21 with no task; coded neither, correct as well-formed; doubt whether counting makes sense in itself.
53435a2cbd54: repeats "The moon landing was a hoax"; coded intended (explanation in degenerate form), incorrect; doubt whether a bare repeated claim does the task at all.


## Batch 05

5bf3cf467050 - Watermelon in the red-fruit list is red inside but green outside; coded correct on the red-flesh reading, could be incorrect for an ordinary reader.
ce214a55c2a7 - "you are not guilty until proven guilty" is a slip for "innocent"; coded correct because the information is right, but the phrasing is sloppy.
99bb8ecedc58 - Papaya and cherimoya are described as red when ripe, which is doubtful; coded incorrect with two of five items wrong, but the call is close.
0741d7434f37 - Opens by completing the prompt with a color list, then drifts into meta text and a Python list of milk; coded neither (no task done, no new task of the same form), so the literal reading is possible.
eb371c571ec9 - Repeats the phrase "Reluctant to do something" as a bullet list, not sentences; coded intended with incorrect, on the reading that a repetitive attempt at the task is still the task.
b6dbbf0d4a8a - Repeats its one sentence once after an "Explanation:" header; coded incorrect as running on, though a single repeat is mild.


## Batch 06

5 33560c2cb759: intended and correct up to the answer; the trailing "Output: what are the three largest oceans on earth?" is an invented question, coded incorrect under the run-on rule, but it is a single line and could be read as correct.
8 3c6638cf30d2: the list names Antarctic Ocean third and calls Indian Ocean third-largest in the same text, and the text is cut off; coded incorrect for the contradiction, but a reader might count the three names as right.
15 68223fa48ba5: "Red fruits include:" stops before any list; coded neither as never reaching the task, though it could be read as a start of the intended task.
18 b192972742fd: "It depends on the situation" could be read as a literal answer to who would wait; the repeated cab advice makes it neither under my reading.
21 52201720f693: it shares the complaint ("I feel like I'm constantly being interrupted") and so is coded intended, but the text is an incoherent loop and could be read as neither.
26 dce6469cfea2: troubleshooting advice implicitly treats the computer as needing repair, but it does not state the point of the complaint; coded neither, correct, with doubt between neither and intended.
29 e8a2d87b893f: it answers the literal question with reasons to make the bed, but opens by disputing the intended claim; coded literal.
37 1614dc77d02b: the meaning given (thank you) is right; "Portuguese verb" is a grammatical mislabel that an ordinary reader would not catch; coded correct.
38 8b86855fafa1: a bare echo of the word, which neither translates nor asks for the language; coded neither and correct because it is well formed, but this is borderline.
35 1aea14d9cbbe: the snowfall minimum in the definition of a blizzard is doubtful, but not obviously wrong to an ordinary reader; coded correct.


## Batch 07

0f3e44dfc3d9: correct chosen; the answer is right, but the elaboration places a distance figure "during a lunar eclipse", which may count as obviously wrong.
4dcb2e17d371: intended and correct chosen only because the item's reference reading treats "Texas is bigger than France in area" as the task; an ordinary reader may judge it wrong.
03815165cab1: incorrect chosen; the first sentence is right, the second ("gravity is relative to the speed of light") makes no sense. Could be read as correct if that sentence is judged loose rather than nonsense.
2659b11125d1: literal chosen (printer and class continue; the requested definition is not given); could be neither. Correctness is doubtful on the ungrammatical "efetivamente efetivamente" sentence.
e7bfb8b0044f: neither chosen; a single echo of the final sentence of the prompt. Could be read as literal (narrative continued).
51d33f4a2c29: neither chosen; the opening is an echo of the narrative sentence followed by a loop of Instruction blocks. Could be read as literal.
ce1560df5925: intended chosen, because the teacher's voice gives a definition-shaped answer; it is a task done only in form. Could be read as literal.
78c96476cec4: intended and incorrect chosen. The frame-story opening is only partly nonsensical; the task is carried out inside it, so correct is possible.
1d8261dd9892: neither chosen; an echo of the whole prompt followed by a loop. Could be read as literal.
fb1cfc1e177b: neither chosen; the loop mentions the printer and a woman, not the climate question. Might be judged as literal if read as the item's narrative.


## Batch 08

c0fa7faf3267: "Parana" is a one-word non-answer that names the state, not the capital; coded neither, incorrect because it does not make sense as a response. Could be read as correct under neither.
f4baaa6a3837: literal by the garbled-return rule; coded incorrect because the ungrammatical repetition of the prompt does not hold together. Borderline on correctness.
330f83420dcc: a printer loop that mentions the child only in passing; coded neither (no task of the same form, little link to the prompt). Could be read as literal.
679f32d737d8: a loop about a child's appearance, with no task and no printer; coded neither, incorrect. Could be read as intended.
93569c0be2b3: the opening claim about photosynthesis is wrong and the text then loops; coded intended, incorrect because the form is an explanation. Could be read as neither.
a90f81863af1: same pattern as 93569c0be2b3 (a wrong definition-like opening, then a loop); coded intended, incorrect. Could be read as neither.
e0f910fe166d: a research-abstract-style text about photosynthesis and blood pressure; coded neither, incorrect. Could be read as intended, incorrect.
36943a0e2b8e: a bulleted list whose items are only the word "Crianças"; coded intended, incorrect on form alone. Could be read as neither.
94f8d627a067: talks about where to find a shopping list and never gives one; coded neither, incorrect. Could be read as intended.
e9ec673d2081: a bulleted list of nonsense items ("Cadeia de..."); coded intended, incorrect on form alone. Borderline with neither.
7266dde19853: a translation given with "Obrigado.", then an invented letter and a cut-off end; coded intended, incorrect. Correctness is clear, but the intent rests on the first line.
01a723adf8c9: a complaint letter mixing Portuguese, Spanish and English with ungrammatical lines; coded intended, incorrect. Borderline on correctness.
44f995e31892: a letter headed "Carta de Defesa" (a defence letter, not a complaint) with a typo in the title; coded intended, incorrect. Could be read as correct.


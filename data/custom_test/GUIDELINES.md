# Custom Egyptian Test Set: Guidelines

This test set answers the research question: **do the models understand Egyptian Arabic?**
It is **evaluation-only**. Nothing here may be used for training or tuning.

## Workflow

1. **Write** sentences into `annotation_sheet.csv`, filling the `id`, `text`, `category` and `domain` columns.
2. **Label independently.** Two annotators fill `label_a1` and `label_a2` without looking at each other's column (hide it while labeling).
3. **Adjudicate.** Where the two labels differ, discuss and put the agreed label in `label`. Leave `label` empty when they already agree.
4. **Write the MSA versions.** For about 150 rows, put an MSA rewrite in `msa_sheet.csv` with the same `id`. The meaning and sentiment must stay the same.
5. **Build** with `python -m src.data.custom_test`. It validates the sheets, reports Cohen's κ, and writes `egy_test.csv` and `msa_parallel.csv`.

## Writing rules

- Write **natural Egyptian Arabic**, the way people actually text, tweet or comment. Don't write MSA with a few Egyptian words sprinkled in.
- **Don't copy** from ASTD, ArSarcasm or any Twitter dataset. The build script rejects any sentence that is also in train, val or test.
- Use **everyday topics**, not politics. The training data is mostly political, so this tests generalization too.
- Keep each sentence to **one clear sentiment**. If a sentence is genuinely mixed ("الأكل حلو بس الخدمة زفت"), only keep it when one side clearly dominates; otherwise drop it.
- Vary length, from 2 words up to about 30.
- Aim for roughly **balanced labels overall**: about 1/3 positive, 1/3 negative, 1/3 neutral.

## Labels

Label the **writer's attitude**, not the topic.

| Label | Meaning | Examples |
|---|---|---|
| `positive` | the writer is pleased, praising or recommending | "الأكل كان تحفة بجد" / "مش وحش خالص" |
| `negative` | the writer is annoyed, complaining or criticizing | "الخدمة زفت" / "تحفة بجد، الطلب وصل بعد ٣ ساعات 👏" |
| `neutral` | a fact, question or announcement, with no attitude | "المحل بيفتح الساعة ١٠" / "حد يعرف النتيجة كام؟" |

Rules:
- **Sarcasm:** label what the writer **means**, not the literal words. "تحفة" said sarcastically is `negative`.
- **Negation:** label the overall meaning. "مش وحش" is `positive`; "مش حلو" is `negative`.
- **Questions:** neutral, unless they're rhetorical complaints ("هو في حد بيشتغل في الشركة دي؟" is `negative`).
- **Wishes and congratulations** ("ربنا يوفقك", "مبروك") are `positive`.

## Categories (`category` column)

Each row gets exactly one category: its **main challenge**.

| Category | Target | What it tests | Example |
|---|---|---|---|
| `egy_vocab` | 80 | Egyptian-only words and slang | "الفيلم ده جامد فشخ" / "الموبايل بايظ" |
| `negation` | 70 | Egyptian negation (مش، ما...ش) that flips sentiment | "ماعجبنيش خالص" / "مش بطال" |
| `sarcasm` | 60 | literal words contradict the meaning | "شكراً يا وي على النت السريع ده 🙃" |
| `intensifier` | 40 | elongation and intensifiers (اوي، خالص، جدااا) | "حلووووو اوي" |
| `emoji` | 40 | the emoji carries or flips the sentiment | "تمام 🙂🔪" / "😍😍😍" |
| `arabizi` | 30 | Egyptian in Latin script / code-switching | "el service kan 7elw bas el as3ar ghalya" |
| `neutral_factual` | 70 | plain facts and questions in Egyptian | "الماتش الساعة ٩ بكرة" |
| `general` | 60 | ordinary Egyptian opinions with no special trick | "المطعم ده حلو والأسعار معقولة" |

Total target: about **450 rows**. `arabizi` is reported separately, because most models have never seen Latin-script Arabic.

## Domains (`domain` column)

Spread rows across: `food`, `movies_tv`, `telecom`, `football`, `shopping`, `transport`, `services`, `daily_life`, `education`, `tech`.

## MSA parallel set (`msa_sheet.csv`)

- Pick about **150 rows** from **all categories except `arabizi`**, weighted toward `egy_vocab`, `negation` and `general`.
- Rewrite each one in clear MSA with the **same meaning and sentiment**. Keep any emoji.
- Example: "الفيلم ده جامد فشخ" → "هذا الفيلم رائع جداً"
- The label is copied automatically from the Egyptian row, so don't relabel.
- The dialect gap is `F1(MSA rows) − F1(matching Egyptian rows)`, computed for each model.

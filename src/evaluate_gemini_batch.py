import pandas as pd
import math
import time
import re
from google import genai
from google.genai import types
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    average_precision_score
)
from tqdm import tqdm

# =======================================
# 1. Gemini API
# =======================================
API_KEY = ""
client = genai.Client(api_key=API_KEY)


# =======================================
# 2. Load Dataset
# =======================================
df_test = pd.read_csv("new_data_test.csv", sep="\t", engine="python")
texts = df_test["reviewContent"].astype(str).tolist()
labels = df_test["flagged"].tolist()


# =======================================
# 3. Prompt Templates
# =======================================

def make_zero_prompt(batch_reviews):
    numbered = "\n".join([f"[{i}] {r}" for i, r in enumerate(batch_reviews)])
    return f"""
You are a classifier for fake reviews.

Task:
- For each review, output ONLY: real or fake
- NOTHING else.

Reviews:
{numbered}

Output format:
[0] real
[1] fake
...
"""


def make_few_shot_prompt(batch_reviews):
    numbered = "\n".join([f"[{i}] {r}" for i, r in enumerate(batch_reviews)])
    return f"""
You are a fake review classifier.

Task:
- For each review, output ONLY: real or fake
- NOTHING else.

Here are examples:

Fake Example 1:
Review: "as new resident chicago los angeles i mission find legit vietnamese food i sick work desperately needed pho delivery asap i stumbled upon simply it s menu promptly ordered pho bo beef pho goi coun shrimp springrolls avocado smoothie for thinks that s gross knock til try it half hour later pho arrive neatly packaged piping hot owner actually threw extra veggies and upgraded smoothie large and gave mango salad free the delivery person said knew i sick wanted feel better wth who that i completely blown away nice called back say thank you now the pho it exactly i needed tasted delicious still good hours later i ate last half it portions great enough 2 skimp beef spring rolls still tasted freshly rolled even delivery too don t forget get extra peanut sauce chili it delish try say hi owner he s awesome"
Label: fake

Fake Example 2:
Review: "back bar makes feel though transported china super cool best pear martinis city great apps too cash only"
Label: fake

Fake Example 3:
Review: "i lunch the gage group 8 this first time there needless say loved it the food better expected given pub food mediocre best the special sandwich braised beef rib sandwich greens swiss thinly sliced granny smith apple pretzel roll big winner the service perfect match food our waiter attentive nice the beer selection outstanding including nice belgian ales well domestic higher end selections the decor nice combination old world feel dark woods modern glass subway tiles we sat rear next kitchen disturbed sounds smells kitchen highly recommend even little pricey side pub lunch"
Label: fake

Fake Example 4:
Review: "just returned costa rica addicted food nuevo leon broke addiction wow basically pay beverages get humungous delicious entree for free they even made something not menu i mentioned dish i love gets vote best mexican food chicago can t wait try everything on menu and though husband thought risky parking baby there hey was parking totally safe too"
Label: fake

Real Example 1:
Review: "great place rarely wait fairly easy find parking expensive order shake order shake they re better real shakes i love everything place except i never decide get everything soooo good i ve heard breakfast mediocre i tried it the caesar portobella wrap bestest thing whole world"
Label: real

Real Example 2:
Review: "hands one favorite tourist restaurants city the food absolutely amazing ambiance inside great i never bad experience always great time whether i go drinks dessert full meal plus take reservations able put name down shop bit return table i love it"
Label: real

Real Example 3:
Review: "hands best chinese i ve ever had i guest china visiting wanted hot pot she actually suggested it i wanted vegetables menu put together gorgeous plate us it came split bowl cooking one spicy soup stock three sauces di instructed add spoon bowl it eat time finished gigantic plate brought us i even think eating more that week ago i ve back since plate baicai quite good i m going back today hunan vegetables it little pricey chinese worth it if go dinner time 7 pm prepared wait line it s busy"
Label: real

Real Example 4:
Review: "all say must go west town minus hideous building food great we went dinner saturday night sat right gazed extensive menu i fancy big eater always little skeptical place boasts mega roll got right hot damn rolls huge before ordered found 13 17dollar rolls tad steep get moneys worth the fish fresh inventive rolls mean puts bacon prosciutto sushi if looking change want step toro coast box give place try byob great qualm proper stemware fan drinking great bottle sake small bucket glass"
Label: real

Now classify the following reviews:
{numbered}

Output ONLY:
[0] real
[1] fake
[2] real
...
"""


def make_cot_prompt(batch_reviews):
    numbered = "\n".join([f"[{i}] {r}" for i, r in enumerate(batch_reviews)])
    return f"""
You are a fake-review classifier.

Task:
- For each review, output ONLY: real or fake
- NOTHING else.

Think step by step based on the following signals:
1. Repetition or marketing-style phrases
2. Overly positive or overly negative tone
3. Lack of specific details about the experience
4. Unnatural language patterns
5. Exaggerated claims 
6. Text length vs meaningful content
7. Rating-text mismatch (if visible)

After analyzing these aspects, decide whether the review is likely real or fake.

But final output MUST be ONLY:

[0] real
[1] fake
...

Reviews:
{numbered}

Answer only with the labels.
"""


# =======================================
# 4. Gemini Safe Call (with retry)
# =======================================
def call_gemini(prompt, model="gemini-2.0-flash"):
    for attempt in range(5):
        try:
            r = client.models.generate_content(
                model=model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )],
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain"
                )
            )
            return r.text
        except Exception as e:
            msg = str(e)
            print("API ERROR:", msg)

            # ---- auto retry for 429 ----
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(" → Auto waiting 45 seconds...")
                time.sleep(45)
                continue

            # other non-retry errors
            return None

    return None  # fail after retries


# =======================================
# 5. Batch Runner (final stable)
# =======================================
def parse_output_to_labels(output, batch_size):
    """
    Strong parser:
    ✓ Parse "[0] real"
    ✓ Parse "0 real"
    ✓ Ignore garbage text
    ✓ Always returns batch_size predictions
    """

    parsed = {}  # index → label

    if output is None:
        return [0] * batch_size

    lines = output.strip().lower().split("\n")

    for line in lines:
        line = line.strip()

        # regex: capture index + (real|fake)
        m = re.match(r".*?(\d+).*?(real|fake)", line)
        if m:
            idx = int(m.group(1))
            lab = m.group(2)
            parsed[idx] = 0 if lab == "real" else 1

    # ensure batch_size predictions
    final_preds = []
    for i in range(batch_size):
        if i in parsed:
            final_preds.append(parsed[i])
        else:
            final_preds.append(0)  # default real

    return final_preds


def run_batch(prompt_fn, out_csv, batch_size=50):
    preds = []
    N = len(texts)
    num_batches = math.ceil(N / batch_size)

    for b in tqdm(range(num_batches), desc=f"Running {out_csv}"):
        batch_reviews = texts[b*batch_size:(b+1)*batch_size]
        # b=0 → texts[0 : 50], b=1 → texts[50 : 100]

        prompt = prompt_fn(batch_reviews)
        output = call_gemini(prompt)

        batch_pred = parse_output_to_labels(output, len(batch_reviews))
        preds.extend(batch_pred)

    # =============================
    # METRICS
    # =============================
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc_roc = roc_auc_score(labels, preds)
    auc_pr = average_precision_score(labels, preds)

    print("\n========== RESULTS ==========")
    print("File:", out_csv)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")

    pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "pred": preds
    }).to_csv(out_csv, index=False)

    return preds


# =======================================
# 6. Run 3 prompts
# =======================================
if __name__ == "__main__":
    run_batch(make_zero_prompt, "gemini_batch_zero_shot.csv")
    run_batch(make_few_shot_prompt, "gemini_batch_few_shot.csv")
    run_batch(make_cot_prompt, "gemini_batch_cot.csv")

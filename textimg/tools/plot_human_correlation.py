import numpy as np
import matplotlib.pyplot as plt


import os
import json


def load_json(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): 
                data.append(json.loads(line))
    return data



models = ["flux2-pro", "gemini-2.5-flash-image", "gpt-image-1.5_low", "gpt-image-1.5", "qwen-image"]

ratio_path = "/n/netscratch/kdbrantley_lab/Lab/jiajunh/text_in_img/render_clear_ratio"
score_path = "/n/netscratch/kdbrantley_lab/Lab/jiajunh/text_in_img/clear_score_label"


scores = []
ratios = []

for model in models:
    scores_dict = {}
    ratios_dict = {}
    pth = os.path.join(score_path, "identical", f"{model}.jsonl")
    print(f"path: {pth}")
    data = load_json(pth)
    for d in data:
        scores_dict[d["id"]] = float(d["score"])

    pth = os.path.join(ratio_path, "identical", f"{model}.jsonl")
    data = load_json(pth)
    for d in data:
        render_quality = d["render_quality"].split(" ")
        ratios_dict[d["id"]] = float(render_quality[0]) / float(render_quality[1])

    for k, v in scores_dict.items():
        scores.append(v / 10.0)
        ratios.append(ratios_dict[k])



for model in models:
    scores_dict = {}
    ratios_dict = {}
    pth = os.path.join(score_path, "reasoning", f"{model}.jsonl")

    data = load_json(pth)
    for d in data:
        scores_dict[d["id"]] = float(d["score"])

    pth = os.path.join(ratio_path, "reasoning", f"{model}.jsonl")
    data = load_json(pth)
    for d in data:
        render_quality = d["render_quality"].split(" ")
        ratios_dict[d["id"]] = float(render_quality[0]) / float(render_quality[1])

    for k, v in scores_dict.items():
        # print(k)
        scores.append(v / 10.0)
        ratios.append(ratios_dict[k])


print(f"original correlation: {np.corrcoef(scores, ratios)[0,1]}")
print(f"original number: {len(scores)}")

print(f"mean score: {np.mean(scores)}, mean ratio: {np.mean(ratios)}")


filter_s = []
filter_r = []

bad_s = []
bad_r = []
for s,r in zip(scores, ratios):
    if not (s < 0.7 and r > 0.9):
        filter_s.append(s)
        filter_r.append(r)
    else:
        bad_s.append(s)
        bad_r.append(r)
print(f"filter correlation: {np.corrcoef(filter_s, filter_r)[0,1]}")
r = np.corrcoef(filter_s, filter_r)[0,1]
print(f"filter number: {len(filter_s)}")



plt.figure(figsize=(6,6))
plt.grid(True, linewidth=0.5, alpha=0.6)
plt.scatter(filter_s, filter_r, color="b", marker='x', s=30, label="Filtered")
plt.scatter(bad_s, bad_r, color="r", marker='x', s=30, label="Remaining")

slope, intercept = np.polyfit(filter_s, filter_r, 1)
x_line = np.linspace(0, 1, 100)
y_line = slope * x_line + intercept

plt.plot(x_line, y_line, color="black", linewidth=1.5, label=f"Pearson r = {r:.3f}")

plt.xlabel("Human Labels")
plt.ylabel("VLM CCR")
plt.title("Correlation between Human Labels and VLM CCR")
plt.legend()
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("scatter.png", dpi=300)
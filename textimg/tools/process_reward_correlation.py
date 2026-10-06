import json
import numpy as np



label_data = []
judge_data = []

with open("./math/overall.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        label = obj["label"]
        judge = obj["judge"]

        avg_judge = sum(judge) / len(judge)
        avg_label = sum(label) / len(label)

        label_data.append(avg_label)
        judge_data.append(avg_judge)

# print(label_data)
# print(judge_data)

corr = np.corrcoef(label_data, judge_data)[0, 1]
print(corr)

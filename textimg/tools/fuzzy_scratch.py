import re
import unicodedata
from rapidfuzz import fuzz

def norm_header(s_list):
    matching = set()
    for s in s_list:
        # s = unicodedata.normalize("NFKC", s)
        s = s.strip()
        # s = s.lower()
        s = re.sub(r"[:]+$", "", s)
        s = re.sub(r"\s+", " ", s)
        matching.add(s)
    return list(matching)

def find_fuzzy_spans(text, keyword, threshold=80):
    # text_n = normalize(text)
    # key_n = normalize(keyword)
    k = len(keyword)

    spans = []
    for i in range(len(text) - k + 1):
        window = text[i:i+k]
        if fuzz.ratio(window, keyword) >= threshold:
            spans.append(text[i:i + k])

    return spans


keyword= "Reasoning"
text = "Reaning: The animal, by carimal, by carying seed on its fur, transports it awa parent plant. This parent plant  allaws land of potenaiblly te land for in the plant in aea new locatong in differents. This dispersal. \nReaoning: Sprouting reffers thern the seed and the new plant, fer the necessibly conditions fo the sessir y for gorely; seeed dispresal it. \nReaoning: Pollen is involveed gemination of ly caride between conditions, like the soil, soil, water, and sunlight )) it merily it merely transportt. \nReaoning: Pollen is invlved, plant reproduction carely plants, or ouscher and Seeds. Seeds, not pollen, so itis not helping with pollintation. \nFertizing: Fertilzing the ground invalceinte to help plants grow. While seed is attached to a way seed fertizilize ground ground around or new parent growth site through fentlization. \nAnswer: A"

print(fuzz.ratio("Reaming", keyword))
spans = find_fuzzy_spans(text, keyword, threshold=80)
print(spans)

matching = norm_header(spans)
print(matching)
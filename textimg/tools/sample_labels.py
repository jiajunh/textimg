import os
import random


if __name__ == "__main__":

    sample_size = 40

    model = "gpt-image-1.5_low"
    # model = "gpt-image-1.5"
    # model = "gemini-2.5-flash-image"
    # model = "flux2-pro"
    # model = "qwen-image"

    length = "64"
    length = "128"
    length = "256"
    length = "512"


    fp = f"/n/netscratch/kdbrantley_lab/Lab/jiajunh/text_in_img/generations/identical/{model}/{length}"
    fp = f"/n/netscratch/kdbrantley_lab/Lab/jiajunh/text_in_img/generations/reasoning/{model}"

    images = os.listdir(fp)
    sample_ids = []

    sample_files = random.sample(images, sample_size)
    for sf in sample_files:
        img_id = int(sf.split(".")[0])
        sample_ids.append(img_id)

    sample_ids = sorted(sample_ids)
    for sid in sample_ids:
        print(f'"id": {sid}, "score: "')

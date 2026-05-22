import json
import random
import argparse

TEST_RATIO = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str)

args = parser.parse_args()

with open(f"{args.data_dir}/transforms.json") as f:
    data = json.load(f)

frames = data["frames"]

random.shuffle(frames)

n_test = int(len(frames) * TEST_RATIO)

test_frames = frames[:n_test]
train_frames = frames[n_test:]

train = dict(data)
test = dict(data)

train["frames"] = train_frames
test["frames"] = test_frames

with open(f"{args.data_dir}/transforms_train.json", "w") as f:
    json.dump(train, f, indent=2)

with open(f"{args.data_dir}/transforms_test.json", "w") as f:
    json.dump(test, f, indent=2)

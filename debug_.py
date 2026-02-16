from datasets import load_dataset


dataset = load_dataset(
    "chengyewang/UMU-bench",
    "Full_Set",
    split="train",
    verification_mode="no_checks"
)

print(dataset[0])
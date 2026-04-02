import os
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SEVERITY_MAP = {
    "blocker":  "Critical",
    "critical": "Critical",
    "major":    "High",
    "normal":   "Medium",
    "minor":    "Low",
    "trivial":  "Low",
}

INTENT_LABELS = sorted([
    "Platform", "JDT", "CDT", "PDE",
    "Bugzilla", "Firefox", "Thunderbird", "Core",
])

SEVERITY_LABELS = ["Critical", "High", "Medium", "Low"]

INTENT_TO_IDX   = {label: i for i, label in enumerate(INTENT_LABELS)}
SEVERITY_TO_IDX = {label: i for i, label in enumerate(SEVERITY_LABELS)}

NUM_INTENT_CLASSES   = len(INTENT_LABELS)
NUM_SEVERITY_CLASSES = len(SEVERITY_LABELS)


def _parse_severity_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for report in root.findall("report"):
        bug_id = report.get("id", "").strip()
        updates = report.findall("update")
        if updates:
            severity = updates[-1].findtext("what", default="").strip().lower()
            rows.append({"bug_id": bug_id, "severity": severity})
    return pd.DataFrame(rows)


def _parse_short_desc_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for report in root.findall("report"):
        bug_id = report.get("id", "").strip()
        updates = report.findall("update")
        if updates:
            text = updates[0].findtext("what", default="").strip()
            rows.append({"bug_id": bug_id, "short_desc": text})
    return pd.DataFrame(rows)


def _load_product(product_dir, product_name):
    severity_path   = os.path.join(product_dir, "severity.xml")
    short_desc_path = os.path.join(product_dir, "short_desc.xml")
    if not os.path.exists(severity_path) or not os.path.exists(short_desc_path):
        return pd.DataFrame()
    severity   = _parse_severity_xml(severity_path)
    short_desc = _parse_short_desc_xml(short_desc_path)
    merged = severity.merge(short_desc, on="bug_id", how="inner")
    merged["product_label"] = product_name
    return merged


def prepare_data(raw_data_dir, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.join(raw_data_dir, "data")

    product_dirs = {
        "Platform":    os.path.join(base, "eclipse", "Platform"),
        "JDT":         os.path.join(base, "eclipse", "JDT"),
        "CDT":         os.path.join(base, "eclipse", "CDT"),
        "PDE":         os.path.join(base, "eclipse", "PDE"),
        "Bugzilla":    os.path.join(base, "mozilla", "Bugzilla"),
        "Firefox":     os.path.join(base, "mozilla", "Firefox"),
        "Thunderbird": os.path.join(base, "mozilla", "Thunderbird"),
        "Core":        os.path.join(base, "mozilla", "Core"),
    }

    frames = []
    for product_name, product_dir in product_dirs.items():
        if not os.path.isdir(product_dir):
            print(f"  Skipping {product_name} - not found: {product_dir}")
            continue
        df = _load_product(product_dir, product_name)
        if not df.empty:
            print(f"  Loaded {len(df):,} bugs from {product_name}")
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data found in {raw_data_dir}.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total raw bugs: {len(df):,}")

    df = df[df["severity"].isin(SEVERITY_MAP.keys())]
    df = df[df["short_desc"].str.strip().str.len() > 5]

    df["severity_label"] = df["severity"].map(SEVERITY_MAP).map(SEVERITY_TO_IDX)
    df["intent_label"]   = df["product_label"].map(INTENT_TO_IDX)
    df = df.dropna(subset=["severity_label", "intent_label"])
    df["severity_label"] = df["severity_label"].astype(int)
    df["intent_label"]   = df["intent_label"].astype(int)
    df["Text"] = df["short_desc"].str.strip()

    min_count = df["intent_label"].value_counts().min()
    print(f"  Balancing to {min_count:,} samples per product")
    balanced = pd.concat([
        group.sample(min_count, random_state=123)
        for _, group in df.groupby("intent_label")
    ]).reset_index(drop=True)

    balanced = balanced.sample(frac=1, random_state=123).reset_index(drop=True)
    n = len(balanced)
    train_end = int(n * 0.70)
    val_end   = train_end + int(n * 0.10)

    train_df = balanced[:train_end]
    val_df   = balanced[train_end:val_end]
    test_df  = balanced[val_end:]

    keep = ["Text", "intent_label", "severity_label"]
    train_df[keep].to_csv(os.path.join(output_dir, "train.csv"),      index=False)
    val_df[keep].to_csv(  os.path.join(output_dir, "validation.csv"), index=False)
    test_df[keep].to_csv( os.path.join(output_dir, "test.csv"),       index=False)

    print(f"\n  Data prepared -> {output_dir}/")
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    print(f"\n  Intent distribution (train):")
    for label, idx in INTENT_TO_IDX.items():
        count = (train_df["intent_label"] == idx).sum()
        print(f"    {label:15s}: {count:,}")

    print(f"\n  Severity distribution (train):")
    for label, idx in SEVERITY_TO_IDX.items():
        count = (train_df["severity_label"] == idx).sum()
        print(f"    {label:10s}: {count:,}")


class CommitDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(str(text)) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [enc[:self.max_length] for enc in self.encoded_texts]
        self.encoded_texts = [
            enc + [pad_token_id] * (self.max_length - len(enc))
            for enc in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.data.iloc[index]["intent_label"],   dtype=torch.long),
            torch.tensor(self.data.iloc[index]["severity_label"], dtype=torch.long),
        )

    def _longest_encoded_length(self):
        return max(len(enc) for enc in self.encoded_texts)


def build_dataloaders(data_dir, tokenizer, batch_size=8, num_workers=0):
    train_dataset = CommitDataset(
        csv_file=os.path.join(data_dir, "train.csv"),
        tokenizer=tokenizer,
        max_length=None,
    )
    val_dataset = CommitDataset(
        csv_file=os.path.join(data_dir, "validation.csv"),
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )
    test_dataset = CommitDataset(
        csv_file=os.path.join(data_dir, "test.csv"),
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader, train_dataset.max_length
"""
The Data Preprocessing Module includes four parts:
1. Parsing XML Reports & Reading Vocabulary
2. Building a Master DataFrame
3. Creating a PyTorch Dataset
4. Main Script
"""

# ===========================================================================================
#  Part 1: Parsing XML Reports & Reading Vocabulary
# ===========================================================================================
import os
import pandas as pd
import xml.etree.ElementTree as ET


def parse_single_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    rows = []

    findings_el = root.find('.//AbstractText[@Label="FINDINGS"]')
    findings_text = findings_el.text.strip() if findings_el is not None and findings_el.text else ""

    impression_el = root.find('.//AbstractText[@Label="IMPRESSION"]')
    impression_text = impression_el.text.strip() if impression_el is not None and impression_el.text else ""

    parent_image_els = root.findall('.//parentImage')

    if len(parent_image_els) == 0:
        rows.append({
            'report_id': "N/A",
            'image_id': None,
            'findings': findings_text,
            'impression': impression_text
        })
    else:
        for img_el in parent_image_els:
            image_id = img_el.get('id', '').strip()
            rows.append({
                'report_id': "N/A",
                'image_id': image_id,
                'findings': findings_text,
                'impression': impression_text
            })

    return rows

def parse_xml_reports(xml_dir):
    all_rows = []
    for fname in os.listdir(xml_dir):
        if fname.endswith(".xml"):
            path = os.path.join(xml_dir, fname)
            all_rows.extend(parse_single_xml(path))
    return all_rows


# ---------------------------------------------------------------------------------------------------------------------
# The below block code is to read the vocab from the document 'radiology_vocabulary_final.csv'
# ---------------------------------------------------------------------------------------------------------------------
def read_vocabulary_synonyms(xlsx_path):
    """
    Reads all columns named 'Term', 'Synonym', 'Synonym...' etc. from the Excel file.
    Builds a dictionary mapping each synonym to the main 'Term'.
    """
    df = pd.read_excel(xlsx_path)

    synonym_cols = [col for col in df.columns if 'Synonym' in col]

    vocab_map = {}

    for _, row in df.iterrows():
        main_term = str(row['Term']).strip()

        for col in synonym_cols:
            val = row[col]
            if pd.notnull(val):
                syn = str(val).strip()
                if syn:
                    vocab_map[syn] = main_term

    return vocab_map

def apply_vocabulary_mapping(text, vocab_map):
    for orig_term, mapped_term in vocab_map.items():
        text = text.replace(orig_term, mapped_term)
    return text


# ===========================================================================================
#  Part 2: Building a Master DataFrame
# ===========================================================================================
def build_master_dataframe(xml_rows, images_folder, vocab_map):
    all_entries = []
    for row in xml_rows:
        img_id = row['image_id']
        if img_id is None:
            image_path = None
        else:
            image_path = os.path.join(images_folder, f"{img_id}.png")
            if not os.path.exists(image_path):
                image_path = None
        findings_mapped = apply_vocabulary_mapping(row['findings'], vocab_map)
        impression_mapped = apply_vocabulary_mapping(row['impression'], vocab_map)
        all_entries.append({
            'report_id': row['report_id'],
            'image_id': img_id,
            'image_path': image_path,
            'findings': findings_mapped,
            'impression': impression_mapped
        })
    return pd.DataFrame(all_entries)


#%%
# ===========================================================================================
#  Part 3: Creating PyTorch Dataset
# ===========================================================================================
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NLMChestXRayDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        findings_text = row['findings'] if pd.notnull(row['findings']) else ""
        impression_text = row['impression'] if pd.notnull(row['impression']) else ""
        return {
            "report_id": row['report_id'],
            "image_id": row['image_id'],
            "image": image,
            "findings": findings_text,
            "impression": impression_text
        }

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ===========================================================================================
#  Part 4: Main Script
# ===========================================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", default="/home/ubuntu/DS_capstone/raw_dataset/test_data/ecgen-radiology")
    parser.add_argument("--images_folder", default="/home/ubuntu/DS_capstone/raw_dataset/test_data/NLMCXR_png")
    parser.add_argument("--vocab_path", default="/home/ubuntu/DS_capstone/raw_dataset/test_data/radiology_vocabulary_final.xlsx")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    xml_rows = parse_xml_reports(args.xml_dir)
    vocab_map = read_vocabulary(args.vocab_path)
    df_master = build_master_dataframe(xml_rows, args.images_folder, vocab_map)
    df_master.to_csv("master_dataset.csv", index=False)

    transform = get_transforms()
    dataset = NLMChestXRayDataset(df_master, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(" - report_ids:", batch_data["report_id"])
        print(" - image_ids:", batch_data["image_id"])
        print(" - image batch shape:", batch_data["image"].shape)
        print(" - sample findings text:", batch_data["findings"][0])
        print(" - sample impression text:", batch_data["impression"][0])
        if batch_idx == 0:
            break

    print("Data preprocessing complete!")

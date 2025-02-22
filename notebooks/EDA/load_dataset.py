"""
Filename: load_dataset.py

Description:
    This module provides functionality to load and index the Indiana University
    Chest X-ray dataset from the Open-i service (National Library of Medicine).
    It defines a PyTorch-compatible Dataset class that pairs each X-ray image
    with its corresponding radiology report.

Usage Example:
    from load_dataset import OpeniChestXrayDataset
    import torch
    from torch.utils.data import DataLoader

    # Example usage:
    dataset = OpeniChestXrayDataset(
        image_dir="/path/to/png_images",
        xml_dir="/path/to/xml_reports"
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for sample in dataloader:
        images = sample["image"]      # raw image or image paths
        reports = sample["report"]    # text reports
        image_ids = sample["image_id"]
"""

#%% -------------------------------------------------------------------------------------------------------------------
"""
    Necessary Packages
"""
import os
import glob
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

#%% -------------------------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

#%% -------------------------------------------------------------------------------------------------------------------
def parse_single_report(xml_file: str) -> Dict[str, Any]:
    """
    Parse an individual XML file to extract relevant report information.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        Dict[str, Any]: A dictionary containing the textual content of the report
                        (e.g., findings, impression) and a list of associated image filenames.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"XML parsing error in file {xml_file}: {e}")
        return {}

    # Basic fields to extract from the XML
    # Adjust tags as needed for your data structure
    findings = ""
    impression = ""
    parent_images = []

    # Example: <MedlineCitation> / <Article> / <Abstract> / <AbstractText Label="FINDINGS">
    # The actual structure may differ depending on the dataset's XML schema
    findings_nodes = root.findall(".//AbstractText[@Label='FINDINGS']")
    if findings_nodes:
        findings = findings_nodes[0].text.strip() if findings_nodes[0].text else ""

    impression_nodes = root.findall(".//AbstractText[@Label='IMPRESSION']")
    if impression_nodes:
        impression = impression_nodes[0].text.strip() if impression_nodes[0].text else ""

    # <parentImage> links image filenames to this report
    image_nodes = root.findall(".//parentImage")
    for img_node in image_nodes:
        img_file = img_node.get("id")  # e.g., 'CXR123.png'
        if img_file:
            parent_images.append(img_file)

    return {
        "findings": findings,
        "impression": impression,
        "parent_images": parent_images
    }


def map_reports_to_images(
    xml_dir: str,
    image_dir: str,
    image_ext: str = ".png"
) -> List[Dict[str, Any]]:
    """
    Traverse XML files in xml_dir, parse each report, and associate with image paths.

    Args:
        xml_dir (str): Directory containing the XML reports.
        image_dir (str): Directory containing the chest X-ray images (PNG or DICOM).
        image_ext (str): Image file extension (e.g., ".png", ".dcm", ".jpg").

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            {
                "image_id": str,
                "image_path": str,
                "report_xml_path": str,
                "findings": str,
                "impression": str
            }
    """
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    logger.info(f"Found {len(xml_files)} XML report files in: {xml_dir}")

    indexed_data = []
    # Create a lookup table for images in image_dir
    # If images are strictly named as in the <parentImage> or "CXRxxx.png",
    # we can match them directly. Otherwise, you may need a more complex approach.
    all_images = set(os.listdir(image_dir))

    for xml_file in xml_files:
        info = parse_single_report(xml_file)
        if not info:
            continue

        for img_filename in info["parent_images"]:
            # Ensure the extension matches (if the dataset uses consistent naming).
            # For example, if the XML references "CXR100.png" but the actual file is "CXR100.PNG",
            # consider normalizing or lowering case. Example below is simplistic.
            if not img_filename.lower().endswith(image_ext.lower()):
                # Replace the extension with the one we expect, or skip
                base_name = os.path.splitext(img_filename)[0]
                candidate = base_name + image_ext
                if candidate in all_images:
                    img_filename = candidate
                else:
                    logger.warning(
                        f"Image file {img_filename} does not match expected extension {image_ext}."
                    )
                    continue

            if img_filename in all_images:
                indexed_data.append({
                    "image_id": os.path.splitext(img_filename)[0],  # e.g., "CXR123"
                    "image_path": os.path.join(image_dir, img_filename),
                    "report_xml_path": xml_file,
                    "findings": info["findings"],
                    "impression": info["impression"]
                })
            else:
                logger.warning(f"Image file {img_filename} not found in {image_dir}.")

    logger.info(f"Successfully mapped {len(indexed_data)} image-report pairs.")
    return indexed_data


class OpeniChestXrayDataset(Dataset):
    """
    A PyTorch Dataset implementation for the Open-i Indiana University Chest X-ray collection.
    This class pairs each X-ray image (PNG or DICOM) with its corresponding radiology report.

    Attributes:
        image_records (List[Dict[str, Any]]): The loaded image-report metadata.
        transform (Optional[callable]): Optional transform to be applied on a sample.
    """
    def __init__(
        self,
        image_dir: str,
        xml_dir: str,
        image_ext: str = ".png",
        transform: Optional[Any] = None,
        load_image: bool = True
    ):
        """
        Constructor for OpeniChestXrayDataset.

        Args:
            image_dir (str): Directory containing the PNG or DICOM images.
            xml_dir (str): Directory containing the XML report files.
            image_ext (str): File extension for the images (e.g., ".png", ".dcm").
            transform (callable, optional): A function/transform to apply to each image sample.
            load_image (bool): Whether to load image data (as PIL Image). If False, __getitem__
                               will return image paths instead of actual image tensors.
        """
        super().__init__()
        self.image_records = map_reports_to_images(xml_dir, image_dir, image_ext=image_ext)
        self.transform = transform
        self.load_image = load_image

    def __len__(self) -> int:
        return len(self.image_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve one sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing:
                {
                    "image": PIL.Image or str,
                    "report": Dict[str, str],
                    "image_id": str
                }
        """
        record = self.image_records[idx]
        sample = {
            "image_id": record["image_id"],
            "report": {
                "findings": record["findings"],
                "impression": record["impression"],
            }
        }

        if self.load_image:
            try:
                # For DICOM files, you might use pydicom or a specialized library to read pixel data
                # For PNG/JPEG, PIL is fine
                image = Image.open(record["image_path"]).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                sample["image"] = image
            except Exception as e:
                logger.error(f"Error loading image {record['image_path']}: {e}")
                sample["image"] = None
        else:
            # Return just the path if user wants to load images later
            sample["image"] = record["image_path"]

        return sample
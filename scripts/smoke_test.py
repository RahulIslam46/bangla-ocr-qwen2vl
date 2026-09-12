"""
Quick local smoke test to verify dataset parsing and image integrity
without downloading large models or exhausting local memory.
"""

import sys
from pathlib import Path
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import parse_bn_htrd, BNHTRdDataset


def run_smoke_test(data_dir: str = "./data/sample"):
    print("=" * 60)
    print(" Running Local Smoke Test for Bangla OCR Setup ")
    print("=" * 60)
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_dir

    print(f"Checking dataset at: {data_path}")
    if not data_path.exists():
        print(f"Error: Path {data_path} does not exist!")
        sys.exit(1)

    # 1. Test Line Mode Parsing
    print("\n[1/3] Testing Line-level Parsing...")
    line_samples = parse_bn_htrd(data_path, mode="line")
    print(f"  --> Found {len(line_samples)} valid line samples.")
    assert len(line_samples) > 0, "No line samples found!"

    first_line = line_samples[0]
    print(f"  First line ID:    {first_line['id']}")
    print(f"  First line image: {first_line['image_path']}")
    print(f"  First line text:  {first_line['text']}")

    # Verify image can be opened
    with Image.open(first_line['image_path']) as img:
        print(f"  Image opened successfully. Format: {img.format}, Size: {img.size}")

    # 2. Test Page Mode Parsing
    print("\n[2/3] Testing Page-level Parsing...")
    page_samples = parse_bn_htrd(data_path, mode="page")
    print(f"  --> Found {len(page_samples)} valid page samples.")
    assert len(page_samples) > 0, "No page samples found!"

    first_page = page_samples[0]
    print(f"  First page ID:    {first_page['id']}")
    print(f"  First page image: {first_page['image_path']}")
    line_count = len(first_page['text'].splitlines())
    print(f"  First page total lines: {line_count}")

    # 3. Test Dataset Class
    print("\n[3/3] Testing PyTorch BNHTRdDataset abstraction...")
    ds = BNHTRdDataset(data_path, mode="line", samples=line_samples[:5])
    print(f"  Dataset initialized with length: {len(ds)}")
    sample_item = ds[0]
    print(f"  Item keys: {list(sample_item.keys())}")
    print(f"  Item text: {sample_item['text']}")
    print(f"  Item image PIL mode: {sample_item['image'].mode}, size: {sample_item['image'].size}")

    print("\n" + "=" * 60)
    print(" ALL LOCAL SMOKE TESTS PASSED! Local setup is verified. ")
    print("=" * 60)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/sample"
    run_smoke_test(data_dir)

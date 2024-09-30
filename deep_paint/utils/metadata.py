import os
import re
import logging
import fire
import pandas as pd
from tqdm import tqdm


def sort_files(item):
    """Sorts files by experiment, plate, and site."""
    # Extract main file name and the number after '_s'
    match = re.search('^(.*?_s)(\d+)$', item)
    if match:
        main_name, num = match.groups()
        return (main_name, int(num))
    return (item, 0)

def generate_metadata(
    image_dir: str,
    image_ext: str,
    save_path: str,
) -> None:
    """
    Generates a CSV file of unique image metadata from a directory of images.

    Parameters
    __________
    :param image_dir: Path to image directory.
    :param image_ext: Image extension (ex: png, jpeg, TIF).
    :param save_path: Path to save metadata csv file to.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Creating save directory...")
    try:
        os.makedirs(save_path, exist_ok=True)
    except FileExistsError:
        pass

    # Need number of files for progress bar
    total_files = sum([len(files) for _, _, files in os.walk(image_dir)])
    pbar = tqdm(os.walk(image_dir), total=total_files, desc="Processing images")

    logging.info("Gathering unique images...")
    data = []
    for root, dirs, files in pbar:
        # Ignore hidden directories
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            # Check if file is an image
            if file.endswith(image_ext):
                # Remove wavelength ("_w") notation and image ext
                base_name = file.rsplit('_w', 1)[0]
                # Extract experiment and plate info from the path
                rel_path = os.path.relpath(root, image_dir)
                components = rel_path.split(os.sep)
                # Image should be contained like: experiment/plate/image.ext
                if len(components) >= 2:
                    experiment, plate = components[:2]
                    data.append({
                        'filename': base_name,
                        'plate': plate,
                        'experiment': experiment}
                    )
            pbar.update(1)

    logging.info("Create metadata csv file...")
    metadata = pd.DataFrame(data)
    metadata = metadata.drop_duplicates()
    metadata = metadata.iloc[metadata['filename'].apply(sort_files).argsort()]
    metadata = metadata.reset_index(drop=True)
    metadata.to_csv(f"{save_path}/metadata.csv", index=False)
    logging.info(f"Saved metadata to: {save_path}/metadata.csv")


if __name__ == "__main__":
    fire.Fire(generate_metadata)

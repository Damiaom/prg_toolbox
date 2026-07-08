import pooch
import os

def get_fmri_data():
    """
    Get the path to the example fMRI data file.

    Returns:
        str: Path to the example fMRI data file.
    """
    # Download and route the file
    # pooch.retrieve downloads the file only if it isn't already there.
    zenodo_url = "https://zenodo.org/records/21133598/files/example_rs-fMRI_data.txt?download=1"
    local_path = pooch.retrieve(
        url=zenodo_url,
        known_hash=None,
        path=".",
        fname="example_rs-fMRI_data.txt"
        )
    print(f"✔ Ready: {local_path}")
    return local_path


def get_spike_data(files='all'):
    """
    Creates a folder and downloads example spike data files into it.

    Returns:
        str: Path to the example spike data folder.
    """
    # Download and route the files
    folder = "./example_data_directory"
    os.makedirs(folder, exist_ok=True)
    zenodo_base_url = "https://zenodo.org/records/21133598/files/"

    gdf_files = [
        "spikes-V4-23E.gdf",
        "spikes-V4-4E.gdf",
        "spikes-V4-5E.gdf",
        "spikes-V4-6E.gdf"
    ]

    if files != 'all':
        gdf_files = [f for f in gdf_files if any(keyword in f for keyword in files)]

    for filename in gdf_files:
        file_url = f"{zenodo_base_url}{filename}?download=1"
        # pooch.retrieve downloads the file only if it isn't already there.
        local_path = pooch.retrieve(
            url=file_url,
            known_hash=None,
            path=folder,
            fname=filename
        )
    print(f"✔ Ready: {local_path}")
    if len(gdf_files) == 1:
        return os.path.join(folder, local_path)
    else:
        return folder
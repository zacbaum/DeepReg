# coding=utf-8

"""
Module to download additional data and resources that
are not included in releases via command line interface.
"""

import argparse
import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def download(dirs, output_dir="./", branch="main"):
    """
    Downloads the files and directories from DeepReg into
    `output_dir`, keeping only `dirs`.

    :param dirs: the list of directories to save
    :param output_dir: directory which we use as the root to save output
    :param branch: The name of the branch from which we download the zip.
    :return: void
    """

    output_dir = os.path.abspath(output_dir)  # Get the output directory.

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("\nWill download folders: [", *dirs, sep=" ", end=" ")
    print("] into {}.\n".format(output_dir))

    zip_url = f"https://github.com/DeepRegNet/DeepReg/archive/{branch}.zip"
    print(f"Downloading archive from DeepReg repository {zip_url}...\n")
    response = urlopen(zip_url)  # Download the zip.
    print("Downloaded archive. Extracting files...\n")

    with ZipFile(BytesIO(response.read())) as zf:

        pathnames = zf.namelist()
        head = pathnames[0]
        keepdirs = [
            os.path.join(head, d) for d in dirs
        ]  # Find our folders to keep, based on what user specifies.

        for pathname in pathnames:
            if any(d in pathname for d in keepdirs):

                info = zf.getinfo(pathname)
                info.filename = info.filename.replace(
                    head, ""
                )  # Remove head directory from filepath
                zf.extract(info, output_dir)

                print("Downloaded {}".format(info.filename))


def main(args=None):
    """
    Function to run in command line with argparse to download data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        "-d",
        dest="output_dir",
        default="./",
        help="All directories will be downloaded to the specified directory.",
    )
    parser.add_argument(
        "--branch",
        "-b",
        dest="branch",
        default="main",
        help="The name of the branch to download.",
    )
    args = parser.parse_args(args)

    dirs = [
        "config",
        "data",
        "demos",
    ]

    download(dirs, args.output_dir, args.branch)

    print(
        "\nDownload complete. "
        "Please refer to the DeepReg Quick Start guide for next steps."
    )


if __name__ == "__main__":
    main()  # pragma: no cover

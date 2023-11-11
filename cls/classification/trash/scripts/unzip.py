import patoolib
import glob
import os

ROOT = "/home/timssh/ML/TAGGING/data"

if __name__ == "__main__":

    file_list = glob.glob(ROOT + f"/*/*.zip")
    file_list.extend(glob.glob(ROOT + f"/*/*.rar"))
    ###########################################################
    file_list.extend(glob.glob(ROOT + f"/*.zip"))
    file_list.extend(glob.glob(ROOT + f"/*.rar"))
    for file in file_list:
        # print('/'.join(file.split('/')[:-1]))
        os.makedirs("/".join(file.split(".")[:-1]), exist_ok=True)
        patoolib.extract_archive(file, outdir="/".join(file.split(".")[:-1]))
        os.remove(file)

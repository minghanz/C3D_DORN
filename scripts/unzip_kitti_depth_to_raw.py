import zipfile
import re
import os
import time
import argparse

"""
This script is to extract new kitti depth dataset (with denser ground truth as depth images) into the corresponding folders of sequences in raw dataset. 
Only extract files of which the corresponding files exists in raw KITTI dataset, otherwise the files are skipped and not extracted. 
This includes train and val data.
Can use this script to extract data_depth_velodyne.zip and data_depth_annotated.zip in KITTI depth dataset. 
"""

##### from unzip.py in monodepth2 repo
##### The test_files.txt in eigen_benchmark split in monodepth2 mostly corresponds to files in val folder in this zip, except that 09_28 images are in train set. 

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, help='path to source zip file to be extracted')
parser.add_argument('-t', '--target', type=str, help='path to unzip the source file to')
args = parser.parse_args()

file_name = args.source
target_folder = args.target
# file_name = '/media/sda1/minghanz/datasets/kitti/data_depth_velodyne.zip'  #"data_download/data_depth_annotated.zip"
# target_folder = "/media/sda1/minghanz/datasets/kitti/kitti_data"

zip_obj = zipfile.ZipFile(file_name)
list_of_files = zip_obj.infolist()

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

not_found_seqs = []
for i,g in enumerate(list_of_files):
    f = g.filename
    if f[-1] == '/':
        continue

    result = re.search("/", f)
    str_needed = f[result.end():]
    # print(str_needed)
    date = str_needed[:10]
    seq_folder = str_needed.split("/")[0]
    seq_path = os.path.join(target_folder, date, seq_folder)

    if os.path.exists(seq_path):
        target_path = os.path.join(target_folder, date, str_needed)
        g.filename = os.path.relpath(target_path, target_folder)

        ## zip_obj.extract can be fed with filename or fileinfo. Here we use fileinfo because we want to modify the path while keeping other information. 
        # zip_obj.extract(f, path=target_path)
        zip_obj.extract(g, path=target_folder)
        if i%100==0:
            print(i, "---------------\n", f, "\n", target_path, "\n")
    else:
        if seq_folder not in not_found_seqs:
            not_found_seqs.append(seq_folder)
            # print("seq {} not found!".format(seq_folder))
            # break
            
with open(os.path.join(target_folder, "readme.txt"), "a") as f:
    ctime = time.ctime()
    ctime = ctime.replace(' ', '_')
    f.write(ctime+'\n')
    f.write("Info by {}: \n".format(__file__))
    f.write("Sequences in kitti depth dataset but not found in kitti raw: \n")
    f.writelines("%s\n"%seq for seq in not_found_seqs)

# # with file_bj.open() as f:
# #     zip_obj.extract("subfile", path="target")
# #     zip_obj.namelist()
import torch
import numpy as np
import os, sys

def get_path(Base_Path,Var):
    file_paths = []
    path = Base_Path + Var + '/'
    dirs = os.listdir(path)
    for subdirs in dirs:
        subpath = path + subdirs + '/'
        subsubdirs = os.listdir(path + subdirs)
        for enddirs in subsubdirs:
            endpath = subpath + enddirs + '/'
            files = os.listdir(subpath + enddirs)
            for file in files:
                epath = endpath + file
                file_paths.append(epath)
    file_paths = np.array(file_paths)
    return file_paths[::12][1::2]

def read_hour_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        ann = f.readlines()
        ann = ann[1:]
        r = []
        for strs in ann:
            strs_list = strs.strip('\n').split(" ")
            r.append(strs_list)
        r = np.array(r)
    return r

def creat_array(file_paths):
    contents = []
    for path in file_paths:
        content = read_hour_txt(path)
        contents.append(content)
    contents = np.array(contents).astype(np.float64)
    return contents

def creat_multivar_array(Base_Path,Var):
    i = 0
    for v in Var:
        file_paths = get_path(Base_Path, v)
        array = torch.tensor(creat_array(file_paths)).transpose(-2, -1)[:, None]
        result = array if i == 0 else torch.cat([result, array], 1)
        i += 1
    return result

Base_Path = './data/'
Var = ['TMP','SHU','PRS']
data = creat_multivar_array(Base_Path,Var)
np.savez('data.npz',data=np.array(data))





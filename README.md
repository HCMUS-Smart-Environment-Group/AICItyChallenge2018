# NVIDIA AI City Challenge. Team 4 - HCMUS

## Introduction

## Usage

## Velocity Freeway

First of all, we need to make necessary folders:
```bash
mkdir bounding_box #Using for storing bounding box information from detector
mdkir video #Using for store video input
mkdir reidentify #Using for storing bounding box and object id after re-identifying
mkdir lines #Using for stroring scanlines information
mkdir velocity #Using for storing velocity results
mkdir submission #Using for storing submission files
```

After, we need to run detector you extract all bounding boxes from video. The bounding boxes information must be saved in Numpy (.npy) format. Each bounding box must be in following format: [x1, y1, x2, y2, score, frame_id]. The filename must be in following format: info_video_name.npy. (For example: info_Loc1_1.npy if video name is Loc1_1.mp4).


## Citation

Please cite our paper in your publications if it helps your research:

+ Title: ***Traffic Flow Analysis with Multiple Adaptive Vehicle Detectors and Velocity Estimation with Landmark-based Scanlines***

+ Author: *Minh-Triet Tran, Tung Dinh-Duy, Thanh-Dat Truong, Vinh Ton-That, Thanh-Nhon Do, Quoc-An Luong, Thanh-Anh Nguyen, Vinh-Tiep Nguyen, Minh Do.*

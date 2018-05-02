# NVIDIA AI City Challenge. Team 4 - HCMUS

## Introduction

In this paper, we propose our method for vehicle detection with multiple adaptive vehicle detectors and velocity estimation with landmark-based scanlines. Inspired by the idea for tiny object detection, we use Faster R-CNN with Resnet-101 to create different specialized vehicle detectors corresponding to different levels of details and poses. We propose a heuristic to check the fitness of a particular vehicle detector to a specific region in camera's view by the mean velocity direction and the mean object size. By this way, we can determine an adaptive set of appropriate vehicle detectors for each region in camera's view. Thus our system is expected to detect vehicles with high accuracy, both in precision and recall, even with tiny objects. 

We exploit the U.S. road rules for the length and distance of broken white lines on roads to propose our method for vehicle's velocity estimation using such landmarks. We determine equally-distributed scanlines, virtual parallel lines that are nearly-perpendicular to the road direction, with reference to the line connecting the corresponding ends of multiple broken white lines. From the timespan for a vehicle to cross two consecutive virtual scanlines, we can calculate the average vehicle's velocity within that road segment. We also refine the speed estimation by detecting when a vehicle stops at a traffic light, and smooth the results with a moving average filter. Experiments on the dataset of Traffic Flow Analysis from NVIDIA AI City Challenge 2018 show that our method achieves the perfect detect rate of $100\%$, the average velocity difference of $6.9762$ mph on freeways, and $8.9144$ mph on both freeways and urban roads.

## Usage

## Multiple Adaptive Vehicle Detectors

## Velocity Freeway

## Velocity Urban

## Citation

Please cite our paper in your publications if it helps your research:

+ Title: ***Traffic Flow Analysis with Multiple Adaptive Vehicle Detectors and Velocity Estimation with Landmark-based Scanlines*** in Proceeding IEEE/CVF Conference Computer Vision and Pattern Recognition Workshops (CVPRW), June 2018.

+ Author: *Minh-Triet Tran, Tung Dinh-Duy, Thanh-Dat Truong, Vinh Ton-That, Thanh-Nhon Do, Quoc-An Luong, Thanh-Anh Nguyen, Vinh-Tiep Nguyen, Minh Do.*

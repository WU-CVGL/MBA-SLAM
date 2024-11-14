<h1 align=center font-weight:100> <strong><i>MBA-SLAM:</i></strong> Motion Blur Aware Dense Visual SLAM with Radiance Fields Representation</h1>

<h3 align="center">arXiv 2024</h3>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-2411.08279-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.08279)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://wangpeng000.github.io/MBA-SLAM/)

## Overview
<p align="center">
    <img src="./assets/MBA-SLAM_compressed.gif" alt="Overview" style="width:75%; height:auto;">
</p>

<div>
Given a sequence of severe motion blurred images and depth, MBA-SLAM can accurately estimate the <strong><i>local camera motion trajectory</i></strong> of each blurred image within the exposure time and recovers the high quality 3D scene.

## Pipeline
<p align="center">
    <img src="./assets/pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
<strong><i>Tracking:</i></strong> Our motion blur-aware tracker directly estimates the camera motion trajectory during the exposure time. <strong><i>Mapping:</i></strong> Our mapper generates virtual sharp images along the camera trajectory.
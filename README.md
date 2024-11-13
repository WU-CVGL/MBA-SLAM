<h1 align=center font-weight:100> <strong><i>MBA-SLAM</i></strong>: Motion Blur Aware Dense Visual SLAM with Radiance Fields Representation</h1>

<h3 align="center">arXiv 2024</h3>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-2411.xxxxx-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.xxxx)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://wangpeng000.github.io/MBA-SLAM/)

## Overview
<p align="center">
    <img src="./assets/teaser1.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
Given a sequence of severe motion blurred images and depth, MBA-SLAM can accurately estimate the local camera motion trajectory of each blurred image within the exposure time and recovers the high quality 3D scene. It surpasses prior vSLAMs on both sharp datasets and motion blurred datasets.

## Pipeline
<p align="center">
    <img src="./assets/pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
<b>Tracking:</b> Given the current blurry frame, the mapper first renders a virtual sharp image of the lastest blurry keyframe from the 3D scene. Our motion blur-aware tracker directly estimates the camera motion trajectory during the exposure time. <b>Mapping:</b> Our mapper generates virtual sharp images along the camera trajectory, following the standard rendering procedures of Radiance Fields or Gaussian Splatting. The blurry image can then be synthesized by averaging these virtual images, adhering to the physical image formation model of motion-blurred images.
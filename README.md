# 📡 A Web-Ready and 5G-Ready Volumetric Video Streaming Platform 📡

## 📝 Overview

This repository accompanies the paper: **[A Web-Ready and 5G-Ready Volumetric Video Streaming Platform: A Platform Prototype and Empirical Study](https://lineainteractividad.github.io/VholoStream/)**
 
This repository accompanies a paper presenting an updated web- and 5G-compatible platform for live volumetric video streaming. It demonstrates real-time transmission and rendering of point-cloud data (PLY/Draco) over WebSocket and HTTP/DASH, and reports on an empirical evaluation of browser and device performance under varied network conditions and transport protocols.


## 🧪 Reproducing the Experiment

To reproduce the experiment:

1. Open the notebook: `metricsanalysis.ipynb`.
2. Follow the steps and code blocks in sequential order.
3. Ensure all required libraries are installed and that `utils.py` is accessible.

This will allow you to analyze and visualize the performance metrics collected from our volumetric video streaming platform under various conditions.


## ✅ Requirements

This Jupyter notebook was created using **Python 3.10.12** and requires the following libraries:

- `pandas==2.2.3`
- `numpy==2.2.0`
- `seaborn==0.13.2`
- `matplotlib==3.9.3`

⚠️ **Note:**  
 - Make sure that `utils.py` is located in the **same folder level** as the notebook.
- The file **all.csv** has to be present in the folder **/experiments/processed/**,this is the processed dataset used in the analisis, all the others are raw datasets.
- The section 1. of the notebook is simply data cleaning, the analysis of the metrics is present in section 2., which starts loading the **all.csv** file to a Pandas DataFrame




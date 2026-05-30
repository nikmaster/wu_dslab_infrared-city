# WU Data Science Lab: infrared.city-1 (Tree Detection)

This repository contains the code and documentation for the **infrared.city-1: Tree detection** project, developed as part of the Data Science Lab at the Vienna University of Economics and Business (WU).

## 🌳 Project Motivation & Overview
As urban areas expand and global temperatures rise due to climate change, understanding how trees can help cool cities, reduce the risk of flooding, and promote health and well-being has become increasingly important.

This project aims to take a step toward better tree coverage in urban areas by building a deep learning model capable of **predicting tree locations using satellite imagery of densely populated areas**.

## 🛰️ Data Sources
1. **Sentinel-2 Satellite Imagery**: Publicly available satellite images with a 10-meter spatial resolution and four color channels (red, green, blue, and near-infrared) provided by the European Space Agency (Copernicus Programme). Imagery from different seasons is used to account for the impact of seasonality on the appearance of trees.
2. **Baumkataster (Tree Cadastre) Vienna**: A publicly available dataset containing the exact location of every documented tree in the city of Vienna, used to train and test our model on exact tree locations.

## 🧠 Model Architecture
The project utilizes a **U-Net deep learning model**, a type of Convolutional Neural Network (CNN) highly effective for image segmentation tasks. 
- The model consists of a contracting path (feature extraction) and an expanding path (upsampling).
- It uses skip connections and temporal attention algorithms to prioritize meaningful images across seasons.
- The model is trained using Binary Cross-Entropy loss between predicted and actual tree locations.

## 📁 Repository Structure
- `project/`: Contains Jupyter notebooks for Exploratory Data Analysis (EDA), data scraping (Sentinel, OSM, Baumkataster), and the machine learning modeling pipelines (`modelT`, `reference_model`).
- `deliverables/`: Project plans, intermediate and final reports, presentations, and sparring group minutes.
- `meetings/`: Meeting preparation and outputs.
- `timesheets/`: Time tracking for project members.
- `links/`: Useful resources and links used throughout the project.

## 👥 Team
- **Dominik Oberhumer** - Organization & Team Management
- **Moritz Hörmansdorfer** - Data Scientist (Machine Learning Training)
- **Nina Salnikow** - Data Engineer (Fetching and Preparation of Data)
- **Sara Klasová** - Data Scientist (Exploratory Data Analysis)

## 🎓 Support & Supervision
- **Supervisor**: Kavita Surana (WU)
- **Data Coaches**: Oana Taut and Vasiliki Fragkia (infrared.city)

## ⚠️ Scope & Limitations
- The model focuses exclusively on tree detection within city environments in the temperate climate zone.
- Analyses regarding heat regulation or flooding prevention are out of scope.
- Satellite image resolution (10 meters) poses challenges for perfectly reliable detection of single, small trees.

# Fake Product Review Detection
The goal of this project is to develop a system that can accurately classify product reviews as either genuine or fake. This is achieved by training a Support Vector Machine (SVM) model on a labeled dataset of reviews. The project integrates a Gradio interface for user interaction, allowing users to input reviews and receive the model's classification output.<br/>
![image](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/827f8d0b-297f-4ed6-a248-ea8877d56781)

## Table of contents
Introduction<br/>
Objectives<br/>
oneAPI<br/>
oneAPI Optimization<br/>
Proposed System <br/>
Key Components<br/>
Workflow<br/>
Output<br/>
Video Explanation<br/>
Setup<br/>
Results and Discussion<br/>

## Introduction

In the era of online shopping, the reliability of product reviews plays a crucial role in assisting consumers with informed purchasing decisions. However, the proliferation of fake product reviews poses a significant challenge to the authenticity and trustworthiness of online reviews. This research focuses on the development of a robust machine learning-based system for the detection of fake product reviews. The proposed solution contributes to the ongoing efforts in creating a trustworthy online shopping environment by empowering consumers with reliable information and assisting online platforms in maintaining the integrity of their review systems.<br/>

*Project PPT:* [intel project ppt.pptx](https://github.com/V-MuthuKamalesh/Fake-Product-Review/files/14085973/intel.project.ppt.pptx)

*Youtube Video Link:* https://youtube.com/shorts/Y_YkDak-PI8?feature=shared
## Objectives
The objective for fake product review detection is to implement a comprehensive and sophisticated system utilizing machine learning algorithms, and user behavior analysis to accurately identify and filter out deceptive reviews. The system aims to analyze only textual content and user profiles for inconsistencies. Continuous learning mechanisms ensure adaptability to evolving fraudulent tactics, while potential collaboration with online platforms and the integration of user authentication mechanisms further enhance the overall effectiveness of the detection system, fostering a more trustworthy and transparent online review ecosystem.
## oneAPI
![image](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/4c803b66-1a39-45f9-860b-67484fa7766d)

The oneAPI AI Analytics Toolkit [1] is implemented using the oneAPI Data Analytics Library (oneDAL), a powerful machine learning library that helps speed up big data analysis. oneDAL is an extension of Intel® Data Analytics Acceleration Library (DAAL) and is a part of oneAPI. oneAPI is a cross-industry, open, standards-based unified programming model that delivers a common developer experience across accelerator architectures. 

## oneAPI Optimization
![image](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/7ba2d794-4308-4312-8ce5-49ce595c74c0)

 I have used *oneAPI* which provides enhancement to ML models using oneAPI libraries for low-level compute optimizations. Optimizing machine learning models using Intel's OneAPI involves focusing on key areas to enhance overall performance. First and foremost, leverage optimized libraries such as oneDAL to efficiently execute deep learning operations. Implement the Data Parallel C++ (DPC++) programming model to tap into parallel processing capabilities across diverse architectures, including CPUs, GPUs, and accelerators. Explore offloading computationally intensive tasks to GPUs using OneAPI's capabilities, leading to significant speed improvements. Prioritize memory optimization strategies, including data layout optimizations, minimizing data movement, and utilizing cache-friendly data structures. Investigate quantization techniques to reduce precision and computational demands. Additionally, consider model pruning to eliminate unnecessary parameters and connections, thus streamlining computations. Use profiling tools to identify and address performance bottlenecks effectively. Finally, stay updated with the latest software and driver releases and engage with Intel's support resources for targeted guidance in optimizing specific areas of your machine learning workflow.
 
## Proposed System
I have used the Machine Learning Model named Support Vector Machine(*SVM*) which gives more accuracy than other models. Support Vector Machines (SVM) is a powerful machine learning technique employed in the realm of fake product review detection. SVM operates by mapping input data into a high-dimensional space and finding the optimal hyperplane that best separates different classes. In the context of detecting fake reviews, SVM analyzes various features extracted from textual content, such as sentiment, word frequency, and linguistic patterns. By training on labeled datasets that distinguish between genuine and deceptive reviews, SVM learns to generalize and classify unseen data effectively. Its ability to handle high-dimensional feature spaces and nonlinear relationships makes SVM particularly suitable for discerning subtle patterns indicative of fraudulent product feedback. This robust approach aids in enhancing the accuracy and reliability of fake review detection systems, contributing to more trustworthy consumer evaluations in the online marketplace.

# Flow Chart
![WhatsApp Image 2024-01-29 at 21 11 32_e4e25e73](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/6a6b9543-bb39-448e-8323-33f4d75658ff)

## Key Components
### Dataset:

A labeled dataset containing genuine and fake product reviews is used for training the SVM model.<br/>
Each review in the dataset is associated with a binary label indicating whether it is genuine or fake.

### Support Vector Machine (SVM) Model:

SVM is a supervised machine learning algorithm that is well-suited for classification tasks.<br/>
The model is trained on the features extracted from the text of product reviews.<br/>
Features may include word embeddings, bag-of-words representation, or TF-IDF scores.

### Gradio Interface:

Gradio is a Python library that simplifies the development of user interfaces for machine learning models.<br/>
The interface allows users to input product reviews and see the model's classification output in real-time.<br/>
It provides an interactive and user-friendly experience for both input and output.

## Workflow:

### Training Phase:

The SVM model is trained using the labeled dataset, learning the patterns and characteristics of genuine and fake reviews.<br/>
Feature extraction methods are applied to represent the reviews in a format suitable for SVM training.

### Integration with Gradio:

Gradio interface is implemented to take user input in the form of a product review.<br/>
The input is processed and fed into the trained SVM model for classification.<br/>
The model's output, indicating whether the review is genuine or fake, is displayed to the user.

### User Interaction:

Users interact with the Gradio interface by inputting product reviews of their choice.<br/>
The interface provides real-time feedback, instantly displaying the model's classification results.

## Output
### Gradio Interface
#### Genuine Review
![WhatsApp Image 2024-01-04 at 07 53 23_4b932bd3](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/a81e2a77-3fbb-4473-b3be-7a1d95cd83ed)
#### Fake Review
![WhatsApp Image 2024-01-04 at 07 55 17_1ac0ddaf](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/61c821d7-d353-4803-b761-042d4c4f403a)


## Video Explanation
https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/32e84182-befa-4cc2-9fe2-9a3154997dfd

## Setup
1.Open IntelPrjFinal.ipynb file in Jupyter notebook.

2.Upload the dataset DatasetIntel.csv in the Jupyter notebook.

The coding is provided to perform the below in IntelPrjFinal.ipynb:

i.Explore the data

ii.Train the machine learning model

iii.Launch the Gradio interface for real-time predictions

*Before running the code run the following commands to install the necessary dependencies*<br/>
import pandas as pd

pip install scikit-learn

pip install gradio

from sklearnex import patch_sklearn

patch_sklearn()
## Results and Discussion
 ![image](https://github.com/V-MuthuKamalesh/Fake-Product-Review/assets/156059656/4c56aa0c-8b10-4567-b414-162910719d64)

### Accuracy for Various ML Models
Naive Bayes-81%<br/>
SVM-90%<br/>
Logistic Regression-88%<br/>
Random Forest-86%<br/>
Decision Tree-77%<br/>

From the above bar chart its evident that SVM is much better than other models for this dataset as SVM gives the accuracy of 90% where all other models produce accuracy below 90%.The ONEAPI reduced the overall runtime and GPU usage significantly compared to normal platforms.

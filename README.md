# Visual Question Answering with Vision-Language Model
This is the code repository for the Master's project: Visual Question Answering with Vision-Language Model  
The model is built upon the CFR_VQA <https://github.com/aioz-ai/CFR_VQA.git>, we further incorporate PEmixer, 2D-RPE and apply Rotary Embedding on both the language and visual encoders.  
We display the high-level overview of our model:  
<img width="510" height="248" alt="image" src="https://github.com/user-attachments/assets/7ada130b-7041-4740-8e50-358776020391" />

Our ablation study result is shown below:  
<img width="420" height="162" alt="image" src="https://github.com/user-attachments/assets/fcc54253-711e-49d2-8298-5a2819a18c4c" /> <img width="282" height="197" alt="image" src="https://github.com/user-attachments/assets/00a76caa-f5bc-464b-8aef-bf6196ef9907" />


We also compare our new model with other benchmarks on this dataset.  
## Summary
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Training](#training)
- [Pretrained models and Testing](#pretrained-models-and-testing)

## Prerequisites
- Python 3.12.3  
- CUDA 12.1
- Pytorch 2.5
Please install dependence package by run following command:
```bash
pip install -r requirements.txt
```
To train the model, run the train.sh  
To test the model, run the test.sh  

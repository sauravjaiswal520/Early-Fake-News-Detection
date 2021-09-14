Topics and Tools for Social Media Data Mining Project
Submitted By:

            Ambuj Guhey       204101005
            Saurabh Jaiswal   204101052
            Utkarsh Khati     204101058
            Samiksha Agarwal  204101069


# Paper of the source codes released:
Chunyuan Yuan, Qianwen Ma, Wei Zhou, Jizhong Han, Songlin Hu. Early Detection of Fake News by Utilizing the Credibility of News, Publishers, and Users Based on Weakly Supervised Learning, COLING 2020.

# Dependencies:
Gensim==3.7.2

Jieba==0.39

Scikit-learn==0.21.2

Pytorch==1.1.0


# Datasets
The main directory contains the directories of Weibo dataset and two Twitter datasets: twitter15 and twitter16. In each directory, there are:
- twitter15.train, twitter15.dev, and twitter15.test file: This files provide traing, development and test samples in a format like: 'source tweet ID \t source tweet content \t label'
  
- twitter15_graph.txt file: This file provides the source posts content of the trees in a format like: 'source tweet ID \t userID1:weight1 userID2:weight2 ...'  

Pickle files are ready in dataset folder.


If you have cuda goto CUDA Folder and follow below steps.

If no CUDA GPU availability execute the steps of in NOCUDA code.



# Reproduce the experimental results:
1. create an empty directory: checkpoint/
2. run script run.py 
3. Results are stored in log files in Logs Folder for updated code.


# UPDATIONS

We have added updations from our end in model/NeuralNetwork.py file, for further refernece refer report.

# Colab for execution in cuda GPU.

We are adding a google colab notebook link for your reference you can directly execute the code here.
https://colab.research.google.com/drive/1nxCsP6fsAeooqbk8-b938i2Q8Sl0iAdP?usp=sharing

The same notebook is downloaded and provided in submission for your reference.
  

OR if you want to run on your colab account

Just upload dataset on google drive , mount drive with notebook and see the flow of code in our notebook execute in same order, 
you will need GPU Runtime enviornment to run the code,for this  Go to google colab click on runtime ->change enviornment type -> GPU








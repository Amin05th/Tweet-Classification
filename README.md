# Tweet-Classification: Project Overview

- Created a tool that classifies tweet classifications to help twitter identify tweet desaster.
- Downloaded dataset from Kaggle got from youtube video.
- Engineered 3 different recurrent models (rnn, gru, lstm) to test which model performs best.
- Finetuned and optimized all 3 models to reach the best possible model

## Code and Resources Used
- **Python Version:** 3:10
- **Packages:** numpy, pandas pytorch, tensorboard, nltk
- **Kaggle Dataset:** [https://www.kaggle.com/c/nlp-getting-started/overview](https://www.kaggle.com/c/nlp-getting-started/overview)
- **Youtube Video:** [https://www.youtube.com/watch?v=aU8OF0htbTo&t=400s](https://www.youtube.com/watch?v=aU8OF0htbTo&t=400s)

## Kaggle Dataset
With the dataset I downloaded from Kaggle I got following data:

- id
- keywoard
- location
- text

## Data Preprocessing
After downloading the dataset I preprocessed the data using nltk and did the following steps:

- Tokenized the data
- Ignored unnecessery words
- Took the base word using stemming
- Created bag of words

 
## Model Building
First I created a Pytorch Dataset using a random split and a Dataloader. Then I tried all 3 recurrent networks to see which one performs best.

## Training and validation
After Model building I begann with training and validation I used for optimization RMStrop and as criterion CrossEntropyLoss.

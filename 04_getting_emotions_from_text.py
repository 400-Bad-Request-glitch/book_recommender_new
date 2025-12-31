import pandas as pd
books=pd.read_csv("books_with_categories.csv")
#insert huggng face model (short cut directly there in jupiter)
#1:40:00
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
classifier("I love this!")
#dataloop.ai you can use this check how did this model perform at classification during training
#1:42:07
from transformers import pipeline
# classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None,device=0)
classifier("I love this!")
#books["description"]["0"]
books["description"][0]
classifier(books["description"][0])
classifier(books["description"][0].split("."))
sentences=books["description"][0].split(".")
predictions=classifier(sentences)
sentences[0]
predictions[0]
predictions
sorted(predictions[0],key=lambda x:x["label"])
import numpy as np
emotion_labels=["anger","disgust","fear","joy","sadness","surprise","neutral"]
isbn=[]
emotion_scores={label:[] for label in emotion_labels}
def calculate_max_emotion_scores(predictions):
    per_emotion_scores= {label:[] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions=sorted(prediction,key=lambda x:x["label"])
        for index,label in enumerate(emotion_labels):
            #per_emotion_scores[label].append(sorted_predictions[index]["scores"])
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label:np.max(scores) for label,scores in per_emotion_scores.items()}#we have now each description is a dictionary max probability for each  of the emotion labels 
#1:50:12
for i in range(10):
    isbn.append(books["isbn13"][i])
    sentences=books["description"][i].split(".")
    predictions=classifier(sentences)
    max_scores=calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])
#emotion_scores
from tqdm import tqdm
emotion_labels=["anger","disgust","fear","joy","sadness","suprise","neutral"]
isbn=[]
emotion_scores={label:[] for label in emotion_labels}
for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences=books["description"][i].split(".")
    predictions=classifier(sentences)
    max_scores=calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])
emotions_df=pd.DataFrame(emotion_scores)
emotions_df["isbn13"]=isbn
#15246
emotions_df.head()
books=pd.merge(books,emotions_df,on="isbn13")

books.to_csv("books_with_emotions.csv",index=False)


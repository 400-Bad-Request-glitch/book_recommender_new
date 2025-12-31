#38:02-->start 56:00 theory
#text to form that allows to compare how the text is similar are they mathematically
#word embeddings
#suppose the words are queen,king,girl,boy,woman,man,tree
#we put in 3d these all we see in 1 of these dimension girl,boy are similar man,woman are similar so this is have to do something like age
# in the other dimension queen and king are similar and different from other so this is to do something like class aukat
#  in the third class people are similar tree is different so this is to do with personhood
# word2vec
# skipgram architecture
# 
#
# 
# 
#
#
#
#
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
books=pd.read_csv("books_cleaned.csv")

#books["tagged_description"]
#vector search
#isbn -->identifier
#text loader in lang chain doesnt work with pandas dataframe 
#books["tagged_description"].to_csv("tagged_description.txt",sep="\n",index=False,header=False)
books["tagged_description"].to_csv(
    "tagged_description.txt",
    index=False,
    header=False,
    encoding="utf-8"
)

#raw_documents=TextLoader("tagged_description.txt").load()
raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()

#text_splitter=CharacterTextSplitter(chunk_size=0,chunk_overlap=0,seperator="\n")4
text_splitter = CharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=0,
    separator="\n"
)
documents=text_splitter.split_documents(raw_documents)
#building vector database
#db_books=Chroma.from_documents(documents,embedding=OpenAIEmbeddings())
#creating vector database
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents,
    embedding=embedding,
    persist_directory="./chroma_books"
)

#querying now
query="A book to tach children about nature"
docs=db_books.similarity_search(query,k=10)
#books[books["isbn13"]== int(docs[0].page_content.split()[0].strip())]
#isbn = docs[0].page_content.split()[0].strip().strip('"')
#books[books["isbn13"] == int(isbn)]

isbn = docs[0].page_content.split()[0].strip().replace('"', '').replace("'", "")
books[books["isbn13"] == int(isbn)]


def retrieve_semantic_recommendations(query:str,top_k:int=10,)->pd.DataFrame:
    recs=db_books.similarity_search(query,k=50)
    books_list=[]
    for i in range(0,len(recs)):
        books_list+=[int(recs[i].page_content.strip('"').split()[0])]
    return books[books["isbn13"].isin(books_list)].head(top_k)
#retrieve_semantic_recommendations("A book to teach children about nature")
#text classification 
#llm will do text classification it will help us sort those categories into smaller number of groups
#once we have these smaller number of categories we can add to our book reccomender as a filter
#1:11:48
#zero shot classification
#pre trained llm --> category
#how they do this encoder,decoder
#how many categforioes are there 
books["categories"].value_counts().reset_index()
books["categories"].value_counts().reset_index().query("count>50")
books[books["categories"]=="Juvinile Fiction"]

category_mapping = {'Fiction' : "Fiction",
 'Juvenile Fiction': "Children's Fiction",
 'Biography & Autobiography': "Nonfiction",
 'History': "Nonfiction",
 'Literary Criticism': "Nonfiction",
 'Philosophy': "Nonfiction",
 'Religion': "Nonfiction",
 'Comics & Graphic Novels': "Fiction",
 'Drama': "Fiction",
 'Juvenile Nonfiction': "Children's Nonfiction",
 'Science': "Nonfiction",
 'Poetry': "Fiction"}#1:17:23
books["simple_categories"]=books["categories"].map(category_mapping)
#1:18:03
#books[(books["simple_categories"].isna())]
#books[~(books["simple_categories"].isna())]
#zero shot classification we are getting our model from hugging face
#hugging face nlp course --> huggingface.co/learn/nlp-course/en/chapter1/1
#pycharm help you directly insert hugging face models
# Use a pipeline as a high-level helper
from transformers import pipeline
fiction_categories=["Fiction","Nonfiction"]
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=0)
#books.loc[books["simple_categories"]== "Fiction","description"].reset_index(drop=True)[0]
sequence=books.loc[books["simple_categories"]== "Fiction","description"].reset_index(drop=True)[0]
pipe(sequence,fiction_categories)
import numpy as np
max_index=np.argmax(pipe(sequence,fiction_categories)["scores"])
#1:25:39
max_label=pipe(sequence,fiction_categories)["labels"][max_index]
def generate_predictions(sequence,categories):
    # predictions=pipe(sequence,categories)["labels"]
    predictions=pipe(sequence,categories)
    max_index=np.argmax(predictions["scores"])
    max_label=predictions["labels"][max_index]
    #scores=pipe(sequence,categories)["scores"]
    #return predictions,scores
    return max_label
#zero shot classification over description and tell us the book is fiction or non fictoon
#how good is our model 
#classify train and test data
from tqdm import tqdm
actual_cats=[]
predicted_cats=[]
for i in tqdm(range(0,300)):
    sequence=books.loc[books["simple_categories"]=="Fiction","description"].reset_index(drop=True)[i]
    predicted_cats+=[generate_predictions(sequence,fiction_categories)]
    actual_cats+=["Fiction"]
for i in tqdm(range(0,300)):
    sequence=books.loc[books["simple_categories"]=="Nonfiction","description"].reset_index(drop=True)[i]
    predicted_cats+=[generate_predictions(sequence,fiction_categories)]
    actual_cats+=["Nonfiction"]

predictions_df=pd.DataFrame({"actual_categories":actual_cats,"predicted_categories":predicted_cats})
predictions_df.head()
# predictions_df["corrected_prediction"]=np.where(predictions_df["actual_categories"]==predictions_df["predicted_categories"],True,False)to_csv("predictions")  bad code to learn                                                         
predictions_df["correct_prediction"]=(
    np.where(predictions_df["actual_categories"]==predictions_df["predicted_categories"],1,0)
)
predictions_df["correct_prediction"].sum()/len(predictions_df)
isbns=[]
predicted_cats=[]
missing_cats=books.loc[books["simple_categories"].isna(),["isbn13","description"]].reset_index(drop=True)
for i in tqdm(range(0,len(missing_cats))):
    sequence=missing_cats["description"][i]
    predicted_cats+=[generate_predictions(sequence,fiction_categories)]
    #isbns+=[missing_cats["isbns13"][i]]
    isbns += [missing_cats["isbn13"][i]]

#1:31:06
missing_predicted_df=pd.DataFrame({"isbn13":isbns,"predicted_categories":predicted_cats})
#merging to original df
books=pd.merge(books,missing_predicted_df,on="isbn13",how="left")
#books["simple_categories"]=np.where(books["simple_categories"].isna(),books["simple_categories"])
books["simple_categories"] = np.where(
    books["simple_categories"].isna(),
    books["predicted_categories"],
    books["simple_categories"]
)
books=books.drop(columns=["predicted_categories"])
books[books["categories"].str.lower().isin([
    "romance",
    "science fiction",
    "sc fi",
    "fantasy",
    "horror",
    "mystery",
    "thriller",
    "comedy",
    "crime",
    "historical"
])]
books.to_csv("books_with_categories.csv",index=False)
#more text classification
#anger,fear,sadness,disgust,joy,suprise,neutral
#now use llm to classification 
#fine tuning we will use do llm classification
#Roberta mode;
#ENCODER
#ENCODER
#ENCODER
#final layers which do mass word prediction
#we fill throw away those fnal layers which do mass word prediction task and use fine tuning will be now used to predict emotional categories
#now we take small labeled dataset (text and emotions) and use them for training









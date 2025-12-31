import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

books=pd.read_csv('books.csv')
#print(books)
#print(books.isnull())
#data cleaning do more by yourself
#ax=plt.axes()
#sns.heatmap(books.isna().transpose(),cbar=False,annot=True,ax=ax)
#plt.xlabel("Columns")
#plt.ylabel("Missing values")
#plt.show()
books["missing_description"]=np.where(books["description"].isna(),1,0)
books["age_of_book"]=2025-books["published_year"]
columns_of_intrest=["num_pages","age_of_book","missing_description","average_rating"]
correlation_matrix=books[columns_of_intrest].corr(method="spearman") #spearman better with non continuous variables
sns.set_theme(style="white")
plt.figure(figsize=(8,6))
#heatmap=sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm",cbar_kws={"label":"Spearman correlation"})
#plt.show()
book_missing=books[~(books["description"].isna()) &
                   ~(books["num_pages"].isna()) &
                   ~(books["average_rating"].isna()) &
                   ~(books["published_year"].isna())
                   ]
#print(book_missing)
print(book_missing["categories"].value_counts().reset_index().sort_values("count",ascending=False))
#use jupyter ,pycharm lots of atomatic plotters no need to write code
#x-categories y-index,x-categories y-count -->bar chart
#bar chart shows how uneven the distribution is ,long tail problem with book category how do we solve this how we normalize how we do this using llm introduce you to text classification
#description column of this dataset must be meaningful and enough long how long they are 
book_missing["words_in_description"]=book_missing["description"].str.split().str.len() 
#work on various charts easy way and code line chart,histogram,histogram x-->words in description y-->count
#create a cutoff atleast these many words must be in the description
book_missing.loc[book_missing["words_in_description"].between(1,4),"description"] #not helpful description
#now change those stuf that is mentioned like (between1,4) 
book_missing_25_words=book_missing.loc[book_missing["words_in_description"]>=25] #cutoff set 25
print(book_missing_25_words)
book_missing_25_words["title_and_subtitle"]=(
    np.where(book_missing_25_words["subtitle"].isna(),book_missing_25_words["title"],
             book_missing_25_words[["title","subtitle"]].astype(str).agg(": ".join,axis=1))
)
#new description column which has each description with unique identifier
book_missing_25_words["tagged_description"]=book_missing_25_words[["isbn13","description"]].astype(str).agg(" ".join,axis=1)
(
    book_missing_25_words
    .drop(["subtitle","missing_description","age_of_book","words_in_description"],axis=1)
    .to_csv("books_cleaned.csv",index=False)
)


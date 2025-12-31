# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv

# import gradio as gr

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# load_dotenv()

# # --------------------------------------------------
# # LOAD DATA
# # --------------------------------------------------
# books = pd.read_csv("books_with_emotions.csv")

# books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# books["large_thumbnail"] = np.where(
#     books["large_thumbnail"].isna(),
#     "cover-not-found.jpg",
#     books["large_thumbnail"],
# )

# # --------------------------------------------------
# # LOAD EXISTING VECTOR DATABASE
# # --------------------------------------------------
# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_books = Chroma(
#     persist_directory="./chroma_books",
#     embedding_function=embedding
# )

# # --------------------------------------------------
# # SEMANTIC RETRIEVAL
# # --------------------------------------------------
# def retrieve_semantic_recommendations(
#     query: str,
#     category: str = "All",
#     tone: str = "All",
#     initial_top_k: int = 50,
#     final_top_k: int = 16,
# ) -> pd.DataFrame:

#     recs = db_books.similarity_search(query, k=initial_top_k)

#     books_list = [
#         int(rec.page_content.split()[0].replace('"', '').replace("'", ""))
#         for rec in recs
#     ]

#     books_recs = books[books["isbn13"].isin(books_list)]

#     # category filter
#     if category != "All":
#         books_recs = books_recs[
#             books_recs["simple_categories"] == category
#         ]

#     # tone filter
#     if tone == "Happy":
#         books_recs = books_recs.sort_values(by="joy", ascending=False)
#     elif tone == "Surprising":
#         books_recs = books_recs.sort_values(by="surprise", ascending=False)
#     elif tone == "Angry":
#         books_recs = books_recs.sort_values(by="anger", ascending=False)
#     elif tone == "Suspenseful":
#         books_recs = books_recs.sort_values(by="fear", ascending=False)
#     elif tone == "Sad":
#         books_recs = books_recs.sort_values(by="sadness", ascending=False)

#     return books_recs.head(final_top_k)

# # --------------------------------------------------
# # GRADIO CALLBACK
# # --------------------------------------------------
# def recommend_books(query: str, category: str, tone: str):
#     recommendations = retrieve_semantic_recommendations(query, category, tone)
#     results = []

#     for _, row in recommendations.iterrows():
#         description = row["description"]
#         truncated_description = " ".join(description.split()[:30]) + "..."

#         authors_split = row["authors"].split(";")
#         if len(authors_split) > 1:
#             authors_str = ", ".join(authors_split[:-1]) + f", and {authors_split[-1]}"
#         else:
#             authors_str = row["authors"]

#         caption = f"{row['title']} by {authors_str}: {truncated_description}"
#         results.append((row["large_thumbnail"], caption))

#     return results

# # --------------------------------------------------
# # UI
# # --------------------------------------------------
# categories = ["All"] + sorted(
#     books["simple_categories"].dropna().unique()
# )
# tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# with gr.Blocks(theme=gr.themes.Glass()) as app:
#     gr.Markdown("# 📚 Semantic Book Recommender")

#     with gr.Row():
#         user_query = gr.Textbox(
#             label="Describe a book you want",
#             placeholder="e.g. A story about forgiveness and redemption"
#         )
#         category_dropdown = gr.Dropdown(
#             choices=categories,
#             label="Category",
#             value="All"
#         )
#         tone_dropdown = gr.Dropdown(
#             choices=tones,
#             label="Emotional Tone",
#             value="All"
#         )
#         submit_button = gr.Button("Find Recommendations")

#     gr.Markdown("## 🔍 Recommendations")
#     output = gr.Gallery(columns=8, rows=2)

#     submit_button.click(
#         fn=recommend_books,
#         inputs=[user_query, category_dropdown, tone_dropdown],
#         outputs=output
#     )

# # if __name__ == "__main__":
# #     app.launch()
# if __name__ == "__main__":
#     dashboard.launch(server_name="0.0.0.0", server_port=7860)


# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# import os
# import gradio as gr

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# load_dotenv()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
# books = pd.read_csv("books_with_emotions.csv")

# books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# books["large_thumbnail"] = np.where(
#     books["large_thumbnail"].isna(),
#     "cover-not-found.jpg",
#     books["large_thumbnail"],
# )

# --------------------------------------------------
# LOAD EXISTING VECTOR DATABASE
# --------------------------------------------------
# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_books = Chroma(
#     persist_directory="./chroma_books",
#     embedding_function=embedding
# )

# --------------------------------------------------
# SEMANTIC RETRIEVAL
# --------------------------------------------------
# def retrieve_semantic_recommendations(
#     query: str,
#     category: str = "All",
#     tone: str = "All",
#     initial_top_k: int = 50,
#     final_top_k: int = 16,
# ) -> pd.DataFrame:

#     recs = db_books.similarity_search(query, k=initial_top_k)

#     books_list = [
#         int(rec.page_content.split()[0].replace('"', '').replace("'", ""))
#         for rec in recs
#     ]

#     books_recs = books[books["isbn13"].isin(books_list)]

#     # category filter
#     if category != "All":
#         books_recs = books_recs[
#             books_recs["simple_categories"] == category
#         ]

#     # tone filter
#     if tone == "Happy":
#         books_recs = books_recs.sort_values(by="joy", ascending=False)
#     elif tone == "Surprising":
#         books_recs = books_recs.sort_values(by="surprise", ascending=False)
#     elif tone == "Angry":
#         books_recs = books_recs.sort_values(by="anger", ascending=False)
#     elif tone == "Suspenseful":
#         books_recs = books_recs.sort_values(by="fear", ascending=False)
#     elif tone == "Sad":
#         books_recs = books_recs.sort_values(by="sadness", ascending=False)

#     return books_recs.head(final_top_k)

# # --------------------------------------------------
# # GRADIO CALLBACK
# # --------------------------------------------------
# def recommend_books(query: str, category: str, tone: str):
#     recommendations = retrieve_semantic_recommendations(query, category, tone)
#     results = []

#     for _, row in recommendations.iterrows():
#         description = row["description"]
#         truncated_description = " ".join(description.split()[:30]) + "..."

#         authors_split = row["authors"].split(";")
#         if len(authors_split) > 1:
#             authors_str = ", ".join(authors_split[:-1]) + f", and {authors_split[-1]}"
#         else:
#             authors_str = row["authors"]

#         caption = f"{row['title']} by {authors_str}: {truncated_description}"
#         results.append((row["large_thumbnail"], caption))

#     return results

# --------------------------------------------------
# UI
# --------------------------------------------------
# categories = ["All"] + sorted(
#     books["simple_categories"].dropna().unique()
# )
# tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# with gr.Blocks(theme=gr.themes.Glass()) as app:
#     gr.Markdown("# 📚 Semantic Book Recommender")

#     with gr.Row():
#         user_query = gr.Textbox(
#             label="Describe a book you want",
#             placeholder="e.g. A story about forgiveness and redemption"
#         )
#         category_dropdown = gr.Dropdown(
#             choices=categories,
#             label="Category",
#             value="All"
#         )
#         tone_dropdown = gr.Dropdown(
#             choices=tones,
#             label="Emotional Tone",
#             value="All"
#         )
#         submit_button = gr.Button("Find Recommendations")

#     gr.Markdown("## 🔍 Recommendations")
#     output = gr.Gallery(columns=8, rows=2)

#     submit_button.click(
#         fn=recommend_books,
#         inputs=[user_query, category_dropdown, tone_dropdown],
#         outputs=output
#     )

# --------------------------------------------------
# RENDER ENTRYPOINT
# --------------------------------------------------
#if __name__ == "__main__":
  #  app.launch(server_name="0.0.0.0", server_port=7860)
# if __name__ == "__main__":
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=int(os.environ.get("PORT", 7860)),
#         show_error=True
#     )
# if __name__ == "__main__":
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=int(os.environ["PORT"]),
#         prevent_thread_lock=True
#     )
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

#load_dotenv()

# --------------------------------------------------
# LOAD BOOK DATA
# --------------------------------------------------
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# --------------------------------------------------
# LOAD EXISTING VECTOR DB (NO TextLoader, NO Splitter)
# --------------------------------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma(
    persist_directory="./chroma_books",
    embedding_function=embedding
)

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------
def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [
        int(rec.page_content.split()[0].replace('"', '').replace("'", ""))
        for rec in recs
    ]

    books_recs = books[books["isbn13"].isin(books_list)]

    # category filter
    if category != "All":
        books_recs = books_recs[
            books_recs["simple_categories"] == category
        ]

    # tone filter
    if tone == "Happy":
        books_recs = books_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        books_recs = books_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        books_recs = books_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        books_recs = books_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        books_recs = books_recs.sort_values(by="sadness", ascending=False)

    return books_recs.head(final_top_k)

# --------------------------------------------------
# GRADIO CALLBACK
# --------------------------------------------------
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) > 1:
            authors_str = ", ".join(authors_split[:-1]) + f", and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# --------------------------------------------------
# UI
# --------------------------------------------------
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book",
            placeholder="e.g. A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone",
            value="All"
        )
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
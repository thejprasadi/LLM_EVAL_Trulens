
import streamlit as st
#from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
#from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# embeddings=OpenAIEmbeddings(
#     model="text-embedding-ada-002"
# )

st.set_page_config(page_title="Research Application" ,layout="wide", initial_sidebar_state="auto")

st.title('PDF Reader')

template = """Answer the question based only on the following context:

{context}
If you don't know the answer, just say out of scope, don't try to make up an answer.


Question: {question}
"""

prompt=ChatPromptTemplate.from_template(template)

model=ChatOpenAI(model_name="gpt-4-turbo-preview",
                 temperature=0)

output_parser=StrOutputParser()

persist_directory="./vectorstore"

upload_file = st.file_uploader("Choose a PDF file", type="pdf")



if upload_file:
    temp_file="./temp.pdf"

    with open(temp_file,"wb") as file:
      file.write(upload_file.getvalue())
    #   file_name=upload_file.file_name
loader= PyPDFLoader(temp_file)
docs_raw = loader.load()

docs_raw_text = [doc.page_content for doc in docs_raw]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
docs = text_splitter.create_documents(docs_raw_text)

    # st.write(docs)
    # st.write("hi...")

   





embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
   

    

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()

   
)
chain.invoke(qa)

def format_docs(docs):

    format_D="\n\n".join([d.page_content for d in docs])
    

    
    return format_D


st.markdown("""
<style>
    /* Target the buttons inside the Streamlit app to expand to full width */
    .stButton>button {
        width: 100%;
    }
            
    
    
</style>
""", unsafe_allow_html=True)

import streamlit as st




col1, col2, col3 = st.columns([3, 1, 3])


with col1:
    user_input = st.text_area("Enter Your Query Here", height=300)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    submit_btn = st.button("Submit", key="summary_btn")
    






with col3:

    if submit_btn:
      with st.spinner('Submiting Query...'):
        qa = submit(user_input)
        st.text_area("Query Output", value=query_result, height=300, key='result')

    else:
      st.text_area("Result", height=300, key='result')
    



    





#####   Evaluation   #######

from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback
import numpy as np
from trulens_eval import TruChain, Tru

tru=Tru()

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App
context = App.select_context(chain)

from trulens_eval.feedback import Groundedness
grounded = Groundedness(groundedness_provider=OpenAI())
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness])

with tru_recorder as recording:
    llm_response = chain.invoke("what is non accrual loan")

display(llm_response)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head(20)
tru.run_dashboard()

rec = recording.get()


for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
# import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate

import os





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



loader= PyPDFLoader("/Users/thejani/Documents/GenAI/report.pdf")
docs_raw = loader.load()

docs_raw_text = [doc.page_content for doc in docs_raw]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=0)
docs = text_splitter.create_documents(docs_raw_text)




embeddings = OpenAIEmbeddings()
#embeddings = OllamaEmbeddings()
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})

def format_docs(docs):

    format_D="\n\n".join([d.page_content for d in docs])
    

    
    return format_D
   

    

chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| model
| StrOutputParser()

   
)

chain.invoke("What is non accrual loans")



#####   Evaluation   #######


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

## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
To develop an AI-powered PDF chatbot that allows users to upload PDF documents and ask questions related to their content. The system retrieves relevant information using RAG, generates accurate responses using a Groq language model, and provides an interactive interface for easy communication with the uploaded documents.

### DESIGN STEPS:

#### STEP 1:
Set up the Python environment in Visual Studio Code and install the required libraries and API key.

#### STEP 2:

Upload the PDF document and extract its text using PyPDFLoader.

#### STEP 3:

Split the text into chunks, generate embeddings, and store them in the FAISS vector database for information retrieval.

### STEP 4:
Connect the retrieved information with the Groq AI model, create the chatbot interface using Panel, and test it with questions from the PDF.

### PROGRAM:

```
import os
import panel as pn

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# STEP 1: Load API key
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# STEP 2: Create AI model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)

# STEP 3: Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

retriever = None
chat_history = []


# STEP 4: Load PDF and create vector database
def load_pdf(file):

    global retriever

    loader = PyPDFLoader(file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    db = FAISS.from_documents(
        docs, embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    return len(docs)


# STEP 5: Ask questions
def ask_question(question):

    global chat_history

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    template = """
    You are a PDF chatbot.

    Answer using only the PDF context.
    If the answer is not available, say:
    I don't know based on the provided PDF.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    answer = response.content

    chat_history.append((question, answer))

    return answer


# STEP 6: Create chatbot interface
pn.extension()

file_input = pn.widgets.FileInput(
    accept=".pdf"
)

load_button = pn.widgets.Button(
    name="Load PDF",
    button_type="primary"
)

question_input = pn.widgets.TextInput(
    placeholder="Ask a question about your PDF"
)

send_button = pn.widgets.Button(
    name="Send",
    button_type="success"
)

status = pn.pane.Markdown("Upload a PDF first.")

chat_display = pn.Column()


# STEP 7: Load uploaded PDF
def load_document(event):

    global retriever, chat_history

    if file_input.value is None:
        status.object = "Please select a PDF."
        return

    try:
        with open("uploaded.pdf", "wb") as f:
            f.write(file_input.value)

        count = load_pdf("uploaded.pdf")

        chat_history = []
        chat_display.clear()

        status.object = f"PDF loaded! Total chunks: {count}"

    except Exception as e:
        status.object = f"Error: {e}"


# STEP 8: Send question
def send_question(event):

    question = question_input.value.strip()

    if not question:
        return

    if retriever is None:
        status.object = "Please upload a PDF first."
        return

    try:
        answer = ask_question(question)

        chat_display.append(
            pn.pane.Markdown(f"**You:** {question}")
        )

        chat_display.append(
            pn.pane.Markdown(f"**Bot:** {answer}")
        )

        question_input.value = ""

    except Exception as e:
        status.object = f"Error: {e}"


load_button.on_click(load_document)
send_button.on_click(send_question)


# STEP 9: Display application
app = pn.Column(
    "# My PDF Chatbot",
    "Upload your PDF and ask questions.",
    file_input,
    load_button,
    status,
    pn.layout.Divider(),
    question_input,
    send_button,
    chat_display,
    width=700
)

pn.serve(app, show=True)
```

### OUTPUT:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2d7e55ff-6cfb-4b81-a2bd-168ba4904305" />


### RESULT:
The PDF chatbot was successfully developed and tested, generating answers to user questions based on the uploaded PDF using RAG and Groq AI.

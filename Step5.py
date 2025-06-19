import cv2
import pytesseract
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

def summarize(text):
    messages= [
        ("system", "Summarizing text."),
        ("human", "Summarize {text}")
    ]
    return ChatPromptTemplate.from_messages(messages)

def find_ke(text):
    messages= [
        ("system", "Extracting key entities from text."),
        ("human", "Extract key entities from {text}")
    ]
    return ChatPromptTemplate.from_messages(messages)

def combine(summary, key_entities):
    return f"Summary: {summary}\nKeyEntities: {key_entities}"


img1 = cv2.imread("C:\Users\vrmic\Downloads\Step5_Img1.png")
img2 = cv2.imread("C:\Users\vrmic\Downloads\Step5_Img2.png")
img3 = cv2.imread("C:\Users\vrmic\Downloads\Step5_Img3.png")

custom_config = r'--oem 3 --psm 3'

text1 = pytesseract.image_to_string(img1, config=custom_config)
text2 = pytesseract.image_to_string(img2, config=custom_config)
text3 = pytesseract.image_to_string(img3, config=custom_config)

text = text1 + text2 + text3

load_dotenv()

model=ChatOpenAI(model='gpt-4o-mini')

messages= [
    ("system", "Understanding the text and extracting information."),
    ("human", "The text is {text}")
]

template = ChatPromptTemplate.from_messages(messages)

summary_chain = (RunnableSequence(lambda x : summarize(x), model, StrOutputParser()))

ke_chain = (RunnableSequence(lambda x : find_ke(x), model, StrOutputParser()))

chain = (template | model | StrOutputParser() | RunnableParallel(branches ={"summary":summary_chain, "key_ent":ke_chain}) | RunnableLambda(lambda x : combine(x["branches"]["summary"], x["branches"]["key_entities"] )))

print(chain.invoke({"text" : text}))
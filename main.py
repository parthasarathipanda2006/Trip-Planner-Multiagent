from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
import uuid
from agents.router import agent,input_schema
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage,AIMessage

app=FastAPI()

class chatprompt(BaseModel):
    prompt:str
    thread_id:str

@app.post('/chat')
def chat(chat:chatprompt):

    config={
        "configurable":{
            "thread_id":chat.thread_id
        }
    }

    response=agent.invoke({"input_state": input_schema(),
        "message_hist": [
            HumanMessage(content=chat.prompt)
        ]},config=config)
    
    return {"AIMessage":response}


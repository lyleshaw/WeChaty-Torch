from typing import List
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
import time
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from typing import TypeVar, Generic, Type, Any
from xml.etree.ElementTree import fromstring
import xml.etree.cElementTree as ET
from starlette.requests import Request
import sys
from pydantic import BaseModel
import os
import json

from model import get_bsimg_pred

# 启动App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    bsimg: str


@app.get("/")
async def main():
    return RedirectResponse('/docs')

@app.post("/message/")
async def m(msg: Message):
    content = get_bsimg_pred(bsimg=msg.bsimg)
    return {
        'content' : content
    }

# 启动服务
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
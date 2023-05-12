import io

from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

@app.get("/healthcheck/")
def healthcheck():
    return 'Health - OK'

# @app.get("wav/{wav}")
# async def waves(wav):
#

@app.get("/word/{word}")
async def words(word:str):
    return {"message": word}

@app.get("/voice/{voice}")
async def words(word:io.BytesIO):
    #create wave/ogg file
    return {"message": word}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('APP_PORT')))
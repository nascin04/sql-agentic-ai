import os
from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv(find_dotenv())

app = FastAPI(
    title="Query API to SQL Agentic AI",
    description="This is a proof of concept for SQL Agentic AI that implements two endpoints: one with a simple chain and another with an advanced SQL agent."
)

class QueryRequest(BaseModel):
    prompt: str

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_NAME = os.getenv("POSTGRES_DB")

db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@db:5432/{DB_NAME}"

db = SQLDatabase.from_uri(db_uri)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

sql_query_chain = create_sql_query_chain(llm, db)

print(sql_query_chain)

@app.post("/query-simple")
async def execute_simple_query(request: QueryRequest):
    try:
        sql_query = sql_query_chain.invoke({"question": request.prompt})
        print(f"SQL Generated (Chain): {sql_query}")
        result = db.run(sql_query)
        return {
            "type": "chain",
            "prompt": request.prompt,
            "generated_sql": sql_query.strip(),
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

@app.post("/query-agent")
async def execute_agent_query(request: QueryRequest):
    try:
        response = agent_executor.invoke({"input": request.prompt})
        
        final_answer = response.get("output")
        
        return {
            "type": "agent",
            "prompt": request.prompt,
            "result": final_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/healthcheck")
def healthcheck():
    return {"status": "running"}
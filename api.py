from fastapi import FastAPI, HTTPException
from db.postgres import insert_store_data, fetch_store_data

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the CCTV Analytics API"}

@app.post("/createTable")
async def create_table(schema_sql: str):
    try:
        await insert_store_data(schema_sql)
        return {"message": "Table created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/addStore")
async def add_identity(identity: dict):
    try:
        identity_id = await insert_store_data(identity)
        return {"id": identity_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/getStores")
async  def get_identities():
    try:
        return await fetch_store_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
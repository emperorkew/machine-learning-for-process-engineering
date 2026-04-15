import supabase
from dotenv import load_dotenv
import os

#setup database connection
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = supabase.create_client(url, key)

#fetch data from database
response = (
    supabase.table('batch_reactor')
    .select('*')
    .execute()
)

def test():
    print("test from connection.py")
"""
Upload CSV data naar Supabase database.
Start met de eenvoudigste dataset: batch_reactor_yield.csv
"""

import os
from pathlib import Path

# Altijd vanuit project root draaien zodat relatieve paden werken
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Supabase connectie
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Zet SUPABASE_URL en SUPABASE_KEY in je .env bestand")

supabase = create_client(url, key)


def upload_batch_reactor():
    """Upload batch_reactor_yield.csv naar Supabase."""

    df = pd.read_csv("data/batch_reactor_yield.csv")
    print(f"Geladen: {len(df)} rijen, {len(df.columns)} kolommen")
    print(f"Kolommen: {list(df.columns)}")
    print(f"\nVoorbeeld eerste 3 rijen:")
    print(df.head(3).to_string(index=False))

    # Kolomnamen aanpassen naar database-vriendelijke namen (lowercase, geen speciale tekens)
    df.columns = [c.lower() for c in df.columns]

    # Omzetten naar lijst van dictionaries (dit is wat Supabase verwacht)
    records = df.to_dict(orient="records")

    # Upload in batches van 500 (Supabase limiet per request)
    batch_size = 500
    total_uploaded = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        response = supabase.table("batch_reactor").insert(batch).execute()
        total_uploaded += len(batch)
        print(f"  Geupload: {total_uploaded}/{len(records)} rijen")

    print(f"\nKlaar! {total_uploaded} rijen geupload naar tabel 'batch_reactor'")


def verify_upload():
    """Controleer of de data correct is geupload."""

    # Tel rijen
    response = supabase.table("batch_reactor").select("*", count="exact").limit(0).execute()
    print(f"Aantal rijen in database: {response.count}")

    # Haal eerste 5 rijen op
    response = supabase.table("batch_reactor").select("*").limit(5).execute()
    df = pd.DataFrame(response.data)
    print(f"\nEerste 5 rijen uit database:")
    print(df.to_string(index=False))

    # Vergelijk met CSV
    df_csv = pd.read_csv("data/batch_reactor_yield.csv")
    df_csv.columns = [c.lower() for c in df_csv.columns]
    print(f"\nEerste 5 rijen uit CSV:")
    print(df_csv.head(5).to_string(index=False))


if __name__ == "__main__":
    print("=== Upload batch_reactor_yield.csv naar Supabase ===\n")
    upload_batch_reactor()
    print("\n=== Verificatie ===\n")
    verify_upload()

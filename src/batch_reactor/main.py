import connection
import pandas as pd


def fetch_as_dataframe(tabel):
    """Fetch data uit Supabase en retourneer als pandas DataFrame."""
    response = connection.fetch(tabel)
    df = pd.DataFrame(response.data)
    return df


def main():
    df = fetch_as_dataframe("batch_reactor")

    print(f"Shape: {df.shape}")
    print(f"\nKolommen: {list(df.columns)}")
    print(f"\nEerste 5 rijen:")
    print(df.head())
    print(f"\nDatatypes:")
    print(df.dtypes)
    print(f"\nStatistieken:")
    print(df.describe())


if __name__ == '__main__':
    main()
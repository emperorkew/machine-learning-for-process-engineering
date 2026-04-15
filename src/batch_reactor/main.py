import connection
import pandas as pd
import matplotlib.pyplot as plt

def fetch_as_dataframe(tabel):
    """Fetch data uit Supabase en retourneer als pandas DataFrame."""
    response = connection.fetch(tabel)
    df = pd.DataFrame(response.data)
    return df

def plot_data(df):
    """Plot alle numerieke kolommen: histogrammen, correlatie met opbrengst, en boxplot per kwaliteit."""
    numeric_cols = df.select_dtypes(include="number").columns.drop("id", errors="ignore")

    # 1. Histogrammen van alle numerieke variabelen
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flat, numeric_cols):
        ax.hist(df[col], bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(col)
        ax.set_ylabel("Frequentie")
    fig.suptitle("Verdeling procesparameters", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 2. Scatterplots: elke parameter vs opbrengst, gekleurd per kwaliteit
    features = [c for c in numeric_cols if c != "opbrengst_pct"]
    colors = {"Premium": "green", "Standaard": "blue", "Afgekeurd": "red"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, features):
        for kwaliteit, color in colors.items():
            subset = df[df["kwaliteit"] == kwaliteit]
            ax.scatter(subset[col], subset["opbrengst_pct"], c=color, alpha=0.4, s=10, label=kwaliteit)
        ax.set_xlabel(col)
        ax.set_ylabel("opbrengst_pct")
    axes.flat[0].legend()
    fig.suptitle("Procesparameters vs Opbrengst", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 3. Boxplot opbrengst per kwaliteitsklasse
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="opbrengst_pct", by="kwaliteit", ax=ax)
    ax.set_title("Opbrengst per kwaliteitsklasse")
    ax.set_ylabel("Opbrengst (%)")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()

    # 4. Correlatiematrix
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.suptitle("Correlatiematrix", fontsize=14)
    plt.tight_layout()
    plt.show()


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

    plot_data(df)


if __name__ == '__main__':
    main()
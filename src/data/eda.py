from src.data.load_data import load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def basic_eda(df):
    print("First 5 rows")
    print(df.head())
    print("Last 5 rows")
    print(df.tail())
    print("25 to 35 rows")
    print(df.iloc[25:35])
    print("column names")
    print(df.columns)
    print("data types")
    print(df.dtypes)
    print("complete information")
    print(df.info())

    print("min")
    print(df.min())
    print("max")
    print(df.max())
    print("duplicates")
    print(df.duplicated())
    print("null values")
    print(df.isnull().sum())
    print(df["PlacementStatus"].value_counts())
    count = df["PlacementStatus"].value_counts()
    plt.figure(figsize=(6,5))
    plt.bar(count.index,count.values)
    plt.bar(count.index,count.values)
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(r"C:\Users\USER\PycharmProjects\Placementpredictionsystem\app\static\charts\bargraph.png")
    plt.show()
def univariate(df):
    plt.figure(figsize=(6,5))
    plt.hist(df["CGPA"],bins=10)
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"C:\Users\USER\PycharmProjects\Placementpredictionsystem\app\static\charts\histogram.png")
    plt.show()


    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6,5))
    plt.pie(gendercount,labels=gendercount.index,autopct="%1.1f%%,startangle=90)")
    plt.title("Gender Distribution of Pie Chart")
    plt.savefig(r"C:\Users\USER\PycharmProjects\Placementpredictionsystem\app\static\charts\piechart.png")
    plt.show()

def bivariate(df):
    plt.figure(figsize=(6,5))
    plt.scatter(df["CGPA"],df["AptitudeTestScore"])
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("AptitudeTestScore")
    plt.savefig(r"C:\Users\USER\PycharmProjects\Placementpredictionsystem\app\static\charts\cgpa_aptitudescore_scatter.png")
    plt.show()
    plt.close()
    plt.figure(figsize=(6,5))
    placed=df[df["PlacementStatus"]==1]["CGPA"]
    not_placed=df[df["PlacementStatus"]==0]["CGPA"]
    plt.boxplot([placed,not_placed],label=["placed","not placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("PlacementStatus")
    plt.ylabel("CGPA")
    plt.savefig(r"C:\Users\USER\PycharmProjects\Placementpredictionsystem\app\static\charts\CGPA_PlacementStatus_boxplot.png")
    plt.show()
    plt.close()

def multivariate(df):

    data=df[["CGPA","AptitudeTestScore","PlacementStatus"]]
    correlation=data.corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",)

    plt.title("Correlation Heatmap")
    plt.savefig(r"C:\Users\USER\PycharmProjects\Placementpredictionsystem\app\static\charts\correlationmatrix.png")
    plt.show()
    plt.close()









if __name__ == "__main__":
    df=load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)

from pathlib import Path
import pandas as pd

def build_analytics(csv_path):
    raw = pd.read_csv(csv_path)
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    total = len(df)
    churn_yes = int((df["Churn"]=="Yes").sum())
    churn_no = int((df["Churn"]=="No").sum())
    churn_rate = round(churn_yes/total*100,2) if total else 0
    def group_rate(col):
        g=df.groupby(col,dropna=False)["Churn"].apply(lambda x:(x=="Yes").mean()*100).round(2)
        return [{"label":str(k),"rate":float(v)} for k,v in g.items()]
    contract = group_rate("Contract")
    gender = df["gender"].value_counts().to_dict()
    churn_dist = {"Churn":churn_yes,"Retained":churn_no}
    monthly_bins = pd.cut(df["MonthlyCharges"], bins=8, duplicates="drop")
    monthly = df.assign(bucket=monthly_bins).groupby("bucket",observed=False)["Churn"].apply(lambda x:(x=="Yes").mean()*100).dropna()
    tenure_bins = pd.cut(df["tenure"], bins=[-1,12,24,36,48,60,72], labels=["0–12","13–24","25–36","37–48","49–60","61–72"])
    tenure = df.assign(bucket=tenure_bins).groupby("bucket",observed=False)["Churn"].apply(lambda x:(x=="Yes").mean()*100).dropna()
    return {
      "total_customers":total,"churned":churn_yes,"retained":churn_no,"churn_rate":churn_rate,
      "churn_distribution":[{"label":k,"value":v} for k,v in churn_dist.items()],
      "gender_distribution":[{"label":str(k),"value":int(v)} for k,v in gender.items()],
      "contract_churn":contract,
      "monthly_charges":[{"label":str(k),"rate":float(v)} for k,v in monthly.items()],
      "tenure_churn":[{"label":str(k),"rate":float(v)} for k,v in tenure.items()],
      "dataset_rows":len(raw),"clean_rows":total
    }

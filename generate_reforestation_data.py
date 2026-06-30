import pandas as pd
import numpy as np

np.random.seed(42)

municipalities = ["Bunawan", "Talacogon", "San Francisco", "Lianga", "Rosario",
                  "Bislig", "Trento", "Veruela", "Loreto", "Lanuza",
                  "Cortes", "Cagwait", "Barobo", "Hinatuan", "Lingig"]

soil_types = ["Loam", "Clay", "Sandy", "Silt", "Peat"]

n = 1000

statuses = np.random.choice(
    ["Successful", "Moderate", "Failed"],
    size=n,
    p=[0.40, 0.35, 0.25]
)

data = {
    "Project_ID": [f"P{i+1:04d}" for i in range(n)],
    "Year": np.random.randint(2015, 2026, n),
    "Municipality": np.random.choice(municipalities, n),
    "Target_Seedlings": np.random.randint(500, 10000, n),
}

data["Planted_Seedlings"] = (data["Target_Seedlings"] * np.random.uniform(0.7, 1.2, n)).astype(int)

survival_means = {"Successful": 0.78, "Moderate": 0.55, "Failed": 0.32}
survival_stds = {"Successful": 0.10, "Moderate": 0.12, "Failed": 0.14}
data["Survival_Rate"] = [
    np.clip(np.random.normal(survival_means[s], survival_stds[s]), 0.05, 0.98)
    for s in statuses
]

fund_means = {"Successful": 13.8, "Moderate": 13.2, "Failed": 12.5}
data["Funding_PHP"] = [
    int(np.clip(np.random.lognormal(mean=fund_means[s], sigma=0.8) + 30000, 30000, 2500000))
    for s in statuses
]

rainfall_means = {"Successful": 2400, "Moderate": 2000, "Failed": 1800}
data["Rainfall_mm"] = [
    max(300, int(np.random.normal(rainfall_means[s], 600)))
    for s in statuses
]

temp_means = {"Successful": 27.5, "Moderate": 28.5, "Failed": 30.0}
data["Temperature_C"] = [
    np.clip(np.random.normal(temp_means[s], 2.5), 20, 38)
    for s in statuses
]

visit_rates = {"Successful": np.random.negative_binomial(3, 0.25, sum(s == "Successful" for s in statuses)),
               "Moderate": np.random.negative_binomial(2, 0.30, sum(s == "Moderate" for s in statuses)),
               "Failed": np.random.negative_binomial(1, 0.35, sum(s == "Failed" for s in statuses))}
monitoring = []
for s in statuses:
    if s == "Successful":
        monitoring.append(np.random.negative_binomial(3, 0.25))
    elif s == "Moderate":
        monitoring.append(np.random.negative_binomial(2, 0.30))
    else:
        monitoring.append(np.random.negative_binomial(1, 0.35))
data["Monitoring_Visits"] = monitoring

data["Soil_Type"] = np.random.choice(soil_types, n)

pest_data = []
for s in statuses:
    if s == "Successful":
        pest_data.append(max(0, int(np.random.poisson(1.0))))
    elif s == "Moderate":
        pest_data.append(max(0, int(np.random.poisson(2.5))))
    else:
        pest_data.append(max(0, int(np.random.poisson(4.5))))
data["Pest_Incidents"] = pest_data

fire_data = []
for s in statuses:
    if s == "Successful":
        fire_data.append(max(0, int(np.random.poisson(0.2))))
    elif s == "Moderate":
        fire_data.append(max(0, int(np.random.poisson(0.6))))
    else:
        fire_data.append(max(0, int(np.random.poisson(1.8))))
data["Fire_Incidents"] = fire_data

df = pd.DataFrame(data)
df["Project_Status"] = statuses

df.to_csv("data/reforestation_projects_1000.csv", index=False)
print(f"Generated {len(df)} projects")
print(df["Project_Status"].value_counts())
print(df.head())
print("\nSummary stats by status:")
print(df.groupby("Project_Status")[["Survival_Rate", "Funding_PHP", "Monitoring_Visits", "Pest_Incidents", "Fire_Incidents"]].mean())

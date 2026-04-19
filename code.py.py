import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest, ttest_ind, pearsonr
import warnings
warnings.filterwarnings('ignore')

PALETTE = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})
SAVE_KW = dict(dpi=150, bbox_inches="tight")


# Section 1 - Load Dataset and Inspect Structure

df_raw = pd.read_csv(r"C:\Users\Sudhanshu\OneDrive\Desktop\ToolBox Python\project\python datatset.csv")

print("INDIA DISTRICT-LEVEL HEALTH AND NUTRITION SURVEY (2015)")
print(f"Total Rows    : {df_raw.shape[0]}")
print(f"Total Columns : {df_raw.shape[1]}")
print(df_raw.head(3).to_string())
print(df_raw.dtypes)
print(df_raw.isnull().sum())


# Section 2 - Data Dictionary

col_dict = {
    "Country": "Always India - identifies the nation",
    "State": "One of 34 Indian states or Union Territories",
    "District": "Sub-state administrative unit",
    "Year": "Survey year Financial Year 2015",
    "Residence Type": "Rural / Urban / Total combined",
    "Female Pop Ever Attended School": "Percentage of women aged 6 and above who attended school",
    "Women Literate 15 to 49": "Percentage of women aged 15-49 who can read and write",
    "Men Literate 15 to 49": "Percentage of men aged 15-49 who can read and write",
    "Women 10 or More Years Schooling": "Percentage of women with 10 or more years of education",
    "Women Married Before 18": "Percentage of women aged 20-24 married before age 18",
    "Men Married Before 21": "Percentage of men aged 25-29 married before age 21",
    "Teen Mothers 15 to 19": "Percentage of teen girls already mothers or pregnant",
    "Any Family Planning": "Percentage of married women using any contraception method",
    "Modern Family Planning": "Percentage of married women using modern contraception methods",
    "Unmet Need Family Planning": "Percentage of women who want to avoid pregnancy but use no method",
    "ANC First Trimester": "Percentage of mothers who had first prenatal check-up in first trimester",
    "4 or More ANC Visits": "Percentage of mothers with 4 or more antenatal care visits",
    "Institutional Births": "Percentage of babies born in a hospital or health facility",
    "Caesarean Section Rate": "Percentage of deliveries by caesarean section",
    "Fully Immunized 12 to 23 months": "Percentage of children aged 12-23 months fully vaccinated",
    "BCG Vaccine": "Percentage of children who received BCG vaccine at birth",
    "Polio 3 Doses": "Percentage of children who received 3 doses of polio vaccine",
    "Stunted Under 5": "Percentage of children under 5 too short for their age - chronic malnutrition",
    "Wasted Under 5": "Percentage of children under 5 too thin for their height - acute malnutrition",
    "Underweight Under 5": "Percentage of children under 5 who are underweight",
    "Women BMI Below Normal": "Percentage of women who are thin or undernourished",
    "Women Overweight or Obese": "Percentage of women who are overweight or obese",
    "Children Anaemic 6 to 59 months": "Percentage of young children with anaemia",
    "Women Anaemic 15 to 49": "Percentage of women of reproductive age with anaemia",
    "Men Anaemic 15 to 49": "Percentage of men with anaemia",
    "Women High Blood Sugar": "Percentage of women with elevated blood glucose",
    "Women Hypertension": "Percentage of women with high blood pressure",
    "Men High Blood Sugar": "Percentage of men with elevated blood glucose",
    "Men Hypertension": "Percentage of men with high blood pressure",
    "Iodized Salt": "Percentage of households using iodised salt",
    "Health Insurance Coverage": "Percentage of households covered by any health insurance scheme",
    "Electricity Access": "Percentage of households with electricity connection",
    "Improved Water Source": "Percentage of households with access to safe drinking water",
    "Improved Sanitation": "Percentage of households with improved toilet or sanitation facility",
    "Clean Fuel for Cooking": "Percentage of households using LPG or clean fuel instead of biomass",
    "Sex Ratio Females per 1000 Males": "Number of females for every 1000 males in population",
    "Sex Ratio at Birth": "Sex ratio among newborns in the last 5 years",
}

print("\nDATA DICTIONARY")
for key, value in col_dict.items():
    print(f"  {key:<45} : {value}")


# Section 3 - Data Preprocessing

df = df_raw.copy()

rename_map = {
    df.columns[5]: "Female_School_Attendance",
    df.columns[6]: "Iodized_Salt_HH",
    df.columns[7]: "Health_Insurance_HH",
    df.columns[8]: "Women_Literacy",
    df.columns[9]: "Men_Literacy",
    df.columns[10]: "Women_10yr_Schooling",
    df.columns[11]: "Child_Marriage_Women",
    df.columns[12]: "Child_Marriage_Men",
    df.columns[13]: "Teen_Mothers",
    df.columns[14]: "FP_Any_Method",
    df.columns[15]: "FP_Modern_Method",
    df.columns[16]: "Pop_Below_15",
    df.columns[17]: "Female_Sterilization",
    df.columns[18]: "Male_Sterilization",
    df.columns[19]: "IUD_Use",
    df.columns[20]: "Pill_Use",
    df.columns[21]: "Condom_Use",
    df.columns[22]: "Unmet_Need_FP",
    df.columns[23]: "Unmet_Need_FP2",
    df.columns[24]: "HW_Talked_FP",
    df.columns[25]: "FP_Side_Effects_Told",
    df.columns[26]: "ANC_1st_Trimester",
    df.columns[27]: "Sex_Ratio",
    df.columns[28]: "ANC_4Plus_Visits",
    df.columns[29]: "TT_Protection",
    df.columns[30]: "IFA_100Days",
    df.columns[31]: "Full_ANC",
    df.columns[32]: "MCP_Card",
    df.columns[33]: "PNC_Mother",
    df.columns[34]: "JSY_Financial_Aid",
    df.columns[35]: "Out_of_Pocket_Delivery_INR",
    df.columns[36]: "Home_Birth_Checkup_24h",
    df.columns[37]: "PNC_Child",
    df.columns[38]: "Sex_Ratio_Birth",
    df.columns[39]: "Institutional_Births",
    df.columns[40]: "Institutional_Births_Public",
    df.columns[41]: "Home_Birth_Skilled",
    df.columns[42]: "Skilled_Birth_Attendant",
    df.columns[43]: "Caesarean_Rate",
    df.columns[44]: "Caesarean_Private",
    df.columns[45]: "Caesarean_Public",
    df.columns[46]: "Full_Immunization",
    df.columns[47]: "BCG_Vaccine",
    df.columns[48]: "Polio_3Dose",
    df.columns[49]: "Birth_Registration",
    df.columns[50]: "DPT_3Dose",
    df.columns[51]: "Measles_3Dose",
    df.columns[52]: "HepB_3Dose",
    df.columns[53]: "Vitamin_A_Dose",
    df.columns[54]: "Vaccine_Public_Facility",
    df.columns[55]: "Vaccine_Private_Facility",
    df.columns[56]: "Diarrhoea_Prevalence",
    df.columns[57]: "Diarrhoea_ORS",
    df.columns[58]: "Diarrhoea_Zinc",
    df.columns[59]: "Diarrhoea_Health_Facility",
    df.columns[60]: "Electricity_Access",
    df.columns[61]: "ARI_Prevalence",
    df.columns[62]: "ARI_Health_Facility",
    df.columns[63]: "Breastfed_1hr_Birth",
    df.columns[64]: "Exclusive_Breastfeeding",
    df.columns[65]: "Solid_Food_6to8m",
    df.columns[66]: "Adequate_Diet_Breastfeeding",
    df.columns[67]: "Adequate_Diet_Non_BF",
    df.columns[68]: "Adequate_Diet_All",
    df.columns[69]: "Stunting",
    df.columns[70]: "Wasting",
    df.columns[71]: "Improved_Water",
    df.columns[72]: "Severe_Wasting",
    df.columns[73]: "Underweight_Under5",
    df.columns[74]: "Women_BMI_Low",
    df.columns[75]: "Men_BMI_Low",
    df.columns[76]: "Women_Overweight",
    df.columns[77]: "Men_Overweight",
    df.columns[78]: "Child_Anaemia",
    df.columns[79]: "NonPreg_Women_Anaemia",
    df.columns[80]: "Preg_Women_Anaemia",
    df.columns[81]: "Women_Anaemia",
    df.columns[82]: "Improved_Sanitation",
    df.columns[83]: "Men_Anaemia",
    df.columns[84]: "Women_High_BS",
    df.columns[85]: "Women_Very_High_BS",
    df.columns[86]: "Men_High_BS",
    df.columns[87]: "Men_Very_High_BS",
    df.columns[88]: "Women_Mild_HTN",
    df.columns[89]: "Women_Moderate_HTN",
    df.columns[90]: "Women_Severe_HTN",
    df.columns[91]: "Men_Mild_HTN",
    df.columns[92]: "Men_Moderate_HTN",
    df.columns[93]: "Clean_Cooking_Fuel",
    df.columns[94]: "Men_Severe_HTN",
    df.columns[95]: "Women_Cervix_Exam",
    df.columns[96]: "Women_Breast_Exam",
    df.columns[97]: "Women_Oral_Exam",
}
df.rename(columns=rename_map, inplace=True)

print(f"Duplicate rows found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

df.drop(columns=["Country", "Year"], inplace=True)

df["Residence Type"] = df["Residence Type"].astype("category")

total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"Missing values: {total_missing} out of {total_cells} ({100 * total_missing / total_cells:.2f}%)")

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("Filled all numeric missing values with column median")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

df_total = df[df["Residence Type"] == "Total"].copy()
df_rural = df[df["Residence Type"] == "Rural"].copy()
df_urban = df[df["Residence Type"] == "Urban"].copy()

print(f"Subsets created - Total: {len(df_total)}, Rural: {len(df_rural)}, Urban: {len(df_urban)}")


# Section 4 - Summary Statistics

key_cols = [
    "Women_Literacy", "Men_Literacy", "Institutional_Births",
    "Full_Immunization", "Stunting", "Wasting", "Underweight_Under5",
    "Women_Anaemia", "Child_Anaemia", "Child_Marriage_Women",
    "Caesarean_Rate", "Clean_Cooking_Fuel", "Improved_Sanitation",
    "Women_Overweight", "Men_High_BS",
]

stats_df = df_total[key_cols].agg(["mean", "median", "std", "min", "max"]).T
stats_df["range"] = stats_df["max"] - stats_df["min"]
stats_df = stats_df.round(2)

print("\nSUMMARY STATISTICS - District Level Total Residence")
print(stats_df.to_string())


# Section 5 - EDA and Visual Storytelling

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Residence Type Distribution Across Districts", fontsize=13, fontweight="bold", y=1.02)

residence_counts = df["Residence Type"].value_counts()

axes[0].bar(residence_counts.index, residence_counts.values, color=PALETTE[:3], edgecolor="white", linewidth=1.5)
axes[0].set_title("Bar Chart", fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(residence_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].pie(residence_counts.values, labels=residence_counts.index, colors=PALETTE[:3],
            autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Pie Chart", fontweight="bold")

state_counts = df.groupby("State")["Residence Type"].count()
state_counts.sort_values().plot(kind="barh", ax=axes[2], color="#3498db", edgecolor="white")
axes[2].set_title("Records Per State", fontweight="bold")
axes[2].set_xlabel("Record Count")

plt.tight_layout()
plt.show()
plt.close()

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
fig.suptitle("Distribution of Key Health Indicators Across All Districts", fontsize=13, fontweight="bold", y=1.01)

for i, col in enumerate(key_cols):
    data = df_total[col].dropna()
    axes[i].hist(data, bins=25, color=PALETTE[i % len(PALETTE)], edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(), color="black", lw=1.5, ls="--", label="Mean")
    axes[i].axvline(data.median(), color="red", lw=1.5, ls=":", label="Median")
    axes[i].set_title(col.replace("_", " "), fontsize=8, fontweight="bold")
    axes[i].set_xlabel("Percentage", fontsize=7)
    axes[i].set_ylabel("Count", fontsize=7)
    axes[i].tick_params(labelsize=7)

axes[0].legend(fontsize=7)
plt.tight_layout()
plt.show()
plt.close()

compare_cols = [
    "Women_Literacy", "Institutional_Births", "Full_Immunization",
    "Stunting", "Women_Anaemia", "Child_Marriage_Women",
    "Clean_Cooking_Fuel", "Improved_Sanitation", "Electricity_Access",
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
fig.suptitle("Rural vs Urban Health Outcome Comparisons", fontsize=13, fontweight="bold", y=1.01)

df_rv = df[df["Residence Type"].isin(["Rural", "Urban"])]
for i, col in enumerate(compare_cols):
    sns.boxplot(data=df_rv, x="Residence Type", y=col,
                palette={"Rural": "#e74c3c", "Urban": "#3498db", "Total": "#2ecc71"},
                ax=axes[i], width=0.5,
                flierprops=dict(marker='o', markersize=3, alpha=0.5))
    axes[i].set_title(col.replace("_", " "), fontweight="bold", fontsize=9)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Percentage", fontsize=8)

plt.tight_layout()
plt.show()
plt.close()

scatter_pairs = [
    ("Women_Literacy", "Institutional_Births", "Do Educated Women Deliver in Hospitals?"),
    ("Child_Marriage_Women", "Teen_Mothers", "Does Child Marriage Cause Teen Pregnancies?"),
    ("Stunting", "Underweight_Under5", "Are Stunted Children Also Underweight?"),
    ("Women_Anaemia", "Child_Anaemia", "Do Mothers and Children Share Anaemia Burden?"),
    ("Clean_Cooking_Fuel", "Women_BMI_Low", "Does Clean Fuel Link to Better Nutrition?"),
    ("Improved_Sanitation", "Stunting", "Does Better Sanitation Reduce Stunting?"),
]

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()
fig.suptitle("Scatter Plots - Exploring Relationships Between Health Indicators", fontsize=13, fontweight="bold", y=1.01)

color_map = {"Rural": "#e74c3c", "Urban": "#3498db", "Total": "#2ecc71"}
for i, (x_col, y_col, title) in enumerate(scatter_pairs):
    for res, grp in df.groupby("Residence Type"):
        axes[i].scatter(grp[x_col], grp[y_col], color=color_map[res], alpha=0.4, s=15, label=res)
    sub = df_total[[x_col, y_col]].dropna()
    m, b, r, p, _ = stats.linregress(sub[x_col], sub[y_col])
    xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
    axes[i].plot(xr, m * xr + b, color="black", lw=1.5,
                 label=f"r={r:.2f}, p={'<.001' if p < .001 else f'{p:.3f}'}")
    axes[i].set_title(title, fontsize=9, fontweight="bold")
    axes[i].set_xlabel(x_col.replace("_", " "), fontsize=8)
    axes[i].set_ylabel(y_col.replace("_", " "), fontsize=8)
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.show()
plt.close()

nutrition_cols = ["Stunting", "Wasting", "Underweight_Under5", "Women_Anaemia", "Child_Anaemia", "Women_BMI_Low"]
pair_df = df_total[nutrition_cols].dropna()

fig = sns.pairplot(pair_df, diag_kind="kde", corner=True,
                   plot_kws={"alpha": 0.4, "s": 15, "color": "#3498db"},
                   diag_kws={"color": "#e74c3c", "fill": True})
fig.figure.suptitle("Pairplot - Nutrition and Anaemia Variables", fontsize=12, fontweight="bold", y=1.01)
plt.show()
plt.close()

outlier_cols = [
    "Caesarean_Rate", "Out_of_Pocket_Delivery_INR",
    "Wasting", "Women_Anaemia", "Child_Marriage_Women", "Teen_Mothers",
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
fig.suptitle("Outlier Detection Using IQR Method", fontsize=13, fontweight="bold", y=1.01)

outlier_summary = {}
for i, col in enumerate(outlier_cols):
    data = df_total[col].dropna()
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
    outlier_summary[col] = n_out

    bp = axes[i].boxplot(data, patch_artist=True,
                         flierprops=dict(marker='D', color='red', markerfacecolor='red', markersize=5),
                         medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(PALETTE[i])
    axes[i].set_title(f"{col.replace('_', ' ')} - {n_out} Outliers", fontsize=9, fontweight="bold")
    axes[i].set_ylabel("Percentage", fontsize=8)
    axes[i].set_xticks([])

plt.tight_layout()
plt.show()
plt.close()
print("Outlier summary:", outlier_summary)


# Objective 1 - Normality Testing and Distribution Analysis

print("\nOBJECTIVE 1 - NORMALITY TESTING AND DISTRIBUTION ANALYSIS")

norm_cols = [
    "Women_Literacy", "Institutional_Births", "Stunting",
    "Women_Anaemia", "Child_Marriage_Women", "Caesarean_Rate",
]

norm_results = []
for col in norm_cols:
    data = df_total[col].dropna()
    sw_stat, sw_p = shapiro(data[:5000])
    da_stat, da_p = normaltest(data)
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
    skew = data.skew()
    kurt = data.kurtosis()
    norm_results.append({
        "Variable": col.replace("_", " "),
        "Shapiro p": round(sw_p, 4),
        "DAgostino p": round(da_p, 4),
        "KS p": round(ks_p, 4),
        "Skewness": round(skew, 3),
        "Kurtosis": round(kurt, 3),
        "Normal": "Yes" if sw_p > 0.05 else "No",
    })
    print(f"  {col:30} | Shapiro p={sw_p:.4f} | Skew={skew:.2f} | {'NORMAL' if sw_p > 0.05 else 'NOT NORMAL'}")

norm_df = pd.DataFrame(norm_results)
print(norm_df.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle("Q-Q Plots for Normality Testing - Dots on Red Line Indicate Normality",
             fontsize=13, fontweight="bold", y=1.01)

for i, col in enumerate(norm_cols):
    data = df_total[col].dropna().values
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    axes[i].plot(osm, osr, 'o', color=PALETTE[i], alpha=0.5, markersize=4)
    axes[i].plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
    sw_stat, sw_p = shapiro(data[:5000])
    label_color = "green" if sw_p > 0.05 else "red"
    axes[i].set_title(f"{col.replace('_', ' ')}\nShapiro p = {sw_p:.4f}",
                      fontsize=9, fontweight="bold", color=label_color)
    axes[i].set_xlabel("Theoretical Quantiles", fontsize=8)
    axes[i].set_ylabel("Sample Quantiles", fontsize=8)

plt.tight_layout()
plt.show()
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
fig.suptitle("Histogram with KDE and Normal Curve Overlay",
             fontsize=13, fontweight="bold", y=1.01)

for i, col in enumerate(norm_cols):
    data = df_total[col].dropna()
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    norm_y = stats.norm.pdf(x, mu, sigma)

    axes[i].hist(data, bins=25, density=True, color=PALETTE[i], alpha=0.6, edgecolor="white")
    data.plot.kde(ax=axes[i], color="navy", lw=2, label="KDE")
    axes[i].plot(x, norm_y, "r--", lw=2, label="Normal Curve")
    axes[i].set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
    axes[i].legend(fontsize=7)
    axes[i].set_xlabel("Value", fontsize=8)
    axes[i].set_ylabel("Density", fontsize=8)

plt.tight_layout()
plt.show()
plt.close()

# Objective 2 - Hypothesis Testing Using Independent Samples t-Test

print("\nOBJECTIVE 2 - HYPOTHESIS TESTING RURAL VS URBAN t-TEST")

ttest_cols = [
    "Women_Literacy", "Institutional_Births", "Full_Immunization",
    "Stunting", "Women_Anaemia", "Child_Marriage_Women",
    "Clean_Cooking_Fuel", "Improved_Sanitation", "Caesarean_Rate",
    "Women_Overweight",
]

ttest_results = []
for col in ttest_cols:
    rural_data = df_rural[col].dropna()
    urban_data = df_urban[col].dropna()
    t_stat, p_val = ttest_ind(rural_data, urban_data)
    diff = urban_data.mean() - rural_data.mean()
    ttest_results.append({
        "Indicator": col.replace("_", " "),
        "Rural Mean": round(rural_data.mean(), 2),
        "Urban Mean": round(urban_data.mean(), 2),
        "Difference": round(diff, 2),
        "t-statistic": round(t_stat, 3),
        "p-value": round(p_val, 5),
        "Significant": "Yes" if p_val < 0.05 else "No",
    })
    print(f"  {col:35} | t={t_stat:6.2f} | p={p_val:.5f} | {'SIGNIFICANT' if p_val < 0.05 else 'not significant'}")

ttest_df = pd.DataFrame(ttest_results)
print(ttest_df[["Indicator", "Rural Mean", "Urban Mean", "Difference", "p-value", "Significant"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(ttest_cols))
width = 0.35

rural_means = [ttest_results[i]["Rural Mean"] for i in range(len(ttest_cols))]
urban_means = [ttest_results[i]["Urban Mean"] for i in range(len(ttest_cols))]

ax.bar(x - width / 2, rural_means, width, color="#e74c3c", alpha=0.85, edgecolor="white", label="Rural")
ax.bar(x + width / 2, urban_means, width, color="#3498db", alpha=0.85, edgecolor="white", label="Urban")

for i, res in enumerate(ttest_results):
    if res["p-value"] < 0.05:
        y_max = max(rural_means[i], urban_means[i]) + 2
        ax.text(i, y_max, "*", ha="center", fontsize=14, color="black", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", " ") for c in ttest_cols], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean Value Percentage", fontsize=11)
ax.set_title("Rural vs Urban Mean Comparison - Star Indicates Significant Difference (p < 0.05)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#e74c3c" if r["p-value"] < 0.05 else "#bdc3c7" for r in ttest_results]
pvals = [r["p-value"] for r in ttest_results]
labels = [r["Indicator"] for r in ttest_results]

ax.barh(labels, [-np.log10(p) for p in pvals], color=colors, edgecolor="white")
ax.axvline(-np.log10(0.05), color="black", lw=1.5, ls="--")
ax.set_xlabel("-log10 of p-value - Larger value means more significant", fontsize=10)
ax.set_title("Statistical Significance of Rural vs Urban Differences\nRed Bars Cross the Significance Threshold",
             fontsize=12, fontweight="bold")

red_patch = mpatches.Patch(color="#e74c3c", label="Significant p < 0.05")
grey_patch = mpatches.Patch(color="#bdc3c7", label="Not Significant")
ax.legend(handles=[red_patch, grey_patch])
plt.tight_layout()
plt.show()
plt.close()

# Objective 3 - Multicollinearity and Correlation Deep Dive

print("\nOBJECTIVE 3 - MULTICOLLINEARITY AND CORRELATION ANALYSIS")

corr_cols = [
    "Women_Literacy", "Men_Literacy", "Child_Marriage_Women", "Teen_Mothers",
    "Institutional_Births", "Full_Immunization", "Stunting", "Wasting",
    "Underweight_Under5", "Women_Anaemia", "Child_Anaemia",
    "Clean_Cooking_Fuel", "Improved_Sanitation", "Electricity_Access",
    "Women_Overweight", "Men_High_BS", "Caesarean_Rate",
]

corr_matrix = df_total[corr_cols].corr()

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix, mask=mask, ax=ax,
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    square=True, linewidths=0.5, linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Pearson r"}
)
ax.set_title("Full Correlation Heatmap - Green Positive, Red Negative, Darker Means Stronger",
             fontsize=14, fontweight="bold", pad=20)
ax.set_xticklabels([c.replace("_", " ") for c in corr_cols], rotation=40, ha="right", fontsize=8)
ax.set_yticklabels([c.replace("_", " ") for c in corr_cols], rotation=0, fontsize=8)
plt.tight_layout()
plt.show()
plt.close()

corr_pairs = (corr_matrix.where(~mask)
              .stack().reset_index()
              .rename(columns={"level_0": "Var1", "level_1": "Var2", 0: "r"}))
corr_pairs = corr_pairs[corr_pairs["r"].abs() < 1].copy()
corr_pairs["abs_r"] = corr_pairs["r"].abs()

top_pos = corr_pairs.nlargest(10, "r")
top_neg = corr_pairs.nsmallest(10, "r")

print("\nTop 10 Positive Correlations:")
for _, row in top_pos.iterrows():
    print(f"  {row['Var1']:35} vs {row['Var2']:35}  r = {row['r']:.3f}")

print("\nTop 10 Negative Correlations:")
for _, row in top_neg.iterrows():
    print(f"  {row['Var1']:35} vs {row['Var2']:35}  r = {row['r']:.3f}")

top15 = corr_pairs.nlargest(15, "abs_r")

fig, ax = plt.subplots(figsize=(13, 7))
labels = [f"{r['Var1'].replace('_', ' ')}  vs  {r['Var2'].replace('_', ' ')}" for _, r in top15.iterrows()]
colors = ["#2ecc71" if r["r"] > 0 else "#e74c3c" for _, r in top15.iterrows()]

for i, (_, row) in enumerate(top15.iterrows()):
    ax.plot([0, row["r"]], [i, i], color=colors[i], lw=2)
    ax.scatter(row["r"], i, color=colors[i], s=120, zorder=5)
    ax.text(row["r"] + 0.01 * (1 if row["r"] > 0 else -1), i,
            f'{row["r"]:.2f}', va="center", fontsize=8,
            ha="left" if row["r"] > 0 else "right")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="black", lw=1)
ax.set_xlim(-1.15, 1.15)
ax.set_xlabel("Pearson Correlation Coefficient r", fontsize=11)
ax.set_title("Top 15 Strongest Correlations - Green Positive, Red Negative",
             fontsize=12, fontweight="bold")
green_patch = mpatches.Patch(color="#2ecc71", label="Positive Correlation")
red_patch = mpatches.Patch(color="#e74c3c", label="Negative Correlation")
ax.legend(handles=[green_patch, red_patch])
plt.tight_layout()
plt.show()
plt.close()

# Objective 4 - Child Nutrition Inequality Across States

print("\nOBJECTIVE 4 - CHILD NUTRITION INEQUALITY ACROSS STATES")

state_nutrition = df_total.groupby("State")[
    ["Stunting", "Wasting", "Underweight_Under5", "Child_Anaemia"]
].mean().sort_values("Stunting", ascending=False)

fig, ax = plt.subplots(figsize=(14, 12))
bottom = np.zeros(len(state_nutrition))
cols_to_stack = ["Stunting", "Wasting", "Underweight_Under5"]
stack_colors = ["#e74c3c", "#f39c12", "#e67e22"]

for scol, sc in zip(cols_to_stack, stack_colors):
    ax.barh(state_nutrition.index, state_nutrition[scol], left=bottom, color=sc, alpha=0.85,
            label=scol.replace("_", " "))
    bottom += state_nutrition[scol].values

ax.set_xlabel("Cumulative Percentage - Stunting + Wasting + Underweight", fontsize=11)
ax.set_title("Child Malnutrition by State - Sorted by Stunting Rate Worst at Top",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(13, 8))
scatter = ax.scatter(
    state_nutrition["Underweight_Under5"],
    state_nutrition["Stunting"],
    s=state_nutrition["Child_Anaemia"] * 5,
    c=state_nutrition["Wasting"],
    cmap="YlOrRd", alpha=0.7, edgecolors="grey"
)
for state, row in state_nutrition.iterrows():
    ax.annotate(state[:12], (row["Underweight_Under5"], row["Stunting"]),
                textcoords="offset points", xytext=(5, 3), fontsize=6.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Wasting Percentage", fontsize=10)
ax.set_xlabel("Underweight Under 5 Percentage", fontsize=11)
ax.set_ylabel("Stunting Percentage", fontsize=11)
ax.set_title("Malnutrition Bubble Chart by State - Bubble Size Shows Child Anaemia, Colour Shows Wasting",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
plt.close()

# Objective 5 - Womens Empowerment and Maternal Health Nexus

print("\nOBJECTIVE 5 - WOMENS EMPOWERMENT AND MATERNAL HEALTH NEXUS")

df_total["Literacy_Quartile"] = pd.qcut(
    df_total["Women_Literacy"], q=4,
    labels=["Q1 Lowest", "Q2", "Q3", "Q4 Highest"]
)

empower_cols = [
    "Child_Marriage_Women", "Teen_Mothers", "Institutional_Births",
    "ANC_4Plus_Visits", "Full_Immunization", "Stunting",
]

quartile_means = df_total.groupby("Literacy_Quartile", observed=True)[empower_cols].mean()

print("\nMean Outcomes by Womens Literacy Quartile:")
print(quartile_means.round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()
fig.suptitle("Womens Literacy Quartile vs Health Outcomes - Q1 Least Literate, Q4 Most Literate",
             fontsize=13, fontweight="bold", y=1.01)

for i, col in enumerate(empower_cols):
    vals = quartile_means[col]
    axes[i].plot(quartile_means.index, vals, marker="o", color="#3498db", lw=2.5, markersize=10)
    axes[i].fill_between(range(4), vals, alpha=0.15, color="#3498db")
    axes[i].set_xticks(range(4))
    axes[i].set_xticklabels(quartile_means.index, rotation=20, ha="right", fontsize=8)
    axes[i].set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
    axes[i].set_ylabel("Mean Percentage", fontsize=8)
    for j, v in enumerate(vals):
        axes[i].annotate(f"{v:.1f}", (j, v), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=8)

plt.tight_layout()
plt.show()
plt.close()

categories = [c.replace("_", " ") for c in empower_cols]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
palette_radar = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

for qi, quartile in enumerate(quartile_means.index):
    values = quartile_means.loc[quartile, empower_cols].tolist()
    max_vals = quartile_means[empower_cols].max()
    values_norm = [v / max_vals[empower_cols[j]] * 100 for j, v in enumerate(values)]
    values_norm += values_norm[:1]
    ax.plot(angles, values_norm, lw=2, label=quartile, color=palette_radar[qi])
    ax.fill(angles, values_norm, alpha=0.05, color=palette_radar[qi])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=9)
ax.set_title("Radar Chart - Literacy Quartile vs Health Outcomes (Normalised for Comparison)",
             fontsize=11, fontweight="bold", pad=30)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
plt.tight_layout()
plt.show()
plt.close()

# Objective 6 - Double Burden Analysis - Malnutrition vs Non-Communicable Diseases

print("\nOBJECTIVE 6 - DOUBLE BURDEN MALNUTRITION VS NON-COMMUNICABLE DISEASES")

r, p = pearsonr(df_urban["Women_Overweight"].dropna(), df_urban["Women_Anaemia"].dropna())
print(f"Urban: Women Overweight vs Women Anaemia | r={r:.3f} p={p:.4f}")

t, p2 = ttest_ind(df_urban["Women_Overweight"].dropna(), df_rural["Women_Overweight"].dropna())
print(f"Overweight Urban Mean: {df_urban['Women_Overweight'].mean():.2f}%")
print(f"Overweight Rural Mean: {df_rural['Women_Overweight'].mean():.2f}%")
print(f"t={t:.3f}  p={p2:.5f}  Result: {'SIGNIFICANT' if p2 < 0.05 else 'not significant'}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("The Double Burden - Malnutrition and Obesity Coexist in Urban India",
             fontsize=13, fontweight="bold", y=1.01)

ax = axes[0]
ax.scatter(df_urban["Women_Overweight"], df_urban["Women_Anaemia"],
           color="#e67e22", alpha=0.5, s=20, label="Urban Districts")
m_u, b_u, r_u, p_u, _ = stats.linregress(
    df_urban["Women_Overweight"].dropna(), df_urban["Women_Anaemia"].dropna()
)
xr = np.linspace(df_urban["Women_Overweight"].min(), df_urban["Women_Overweight"].max(), 100)
ax.plot(xr, m_u * xr + b_u, "k-", lw=2)
ax.set_xlabel("Women Overweight Percentage", fontsize=11)
ax.set_ylabel("Women Anaemia Percentage", fontsize=11)
ax.set_title(f"Urban Districts - Overweight vs Anaemia\nr={r_u:.2f}, p={p_u:.4f}",
             fontsize=10, fontweight="bold")
ax.legend()

ax2 = axes[1]
cats = ["Stunting", "Women_Anaemia", "Women_Overweight", "Men_High_BS", "Women_Mild_HTN"]
rural_vals = [df_rural[c].mean() for c in cats]
urban_vals = [df_urban[c].mean() for c in cats]
x2 = np.arange(len(cats))
ax2.bar(x2 - 0.2, rural_vals, 0.4, color="#e74c3c", alpha=0.8, label="Rural")
ax2.bar(x2 + 0.2, urban_vals, 0.4, color="#3498db", alpha=0.8, label="Urban")
ax2.set_xticks(x2)
ax2.set_xticklabels([c.replace("_", " ") for c in cats], rotation=25, ha="right", fontsize=9)
ax2.set_ylabel("Mean Percentage", fontsize=11)
ax2.set_title("Malnutrition Indicators vs NCD Risk Indicators by Residence Type",
              fontsize=10, fontweight="bold")
ax2.legend()
plt.tight_layout()
plt.show()
plt.close()

adv_cols = [
    "Women_Literacy", "Institutional_Births", "Clean_Cooking_Fuel",
    "Full_Immunization", "Electricity_Access", "Improved_Sanitation",
    "Stunting", "Women_Anaemia", "Child_Marriage_Women",
    "Women_Overweight", "Men_High_BS", "Women_Mild_HTN",
]
diff_vals = [df_urban[c].mean() - df_rural[c].mean() for c in adv_cols]
labels_d = [c.replace("_", " ") for c in adv_cols]
colors_d = ["#3498db" if d > 0 else "#e74c3c" for d in diff_vals]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(labels_d, diff_vals, color=colors_d, edgecolor="white")
ax.axvline(0, color="black", lw=1.5)
ax.set_xlabel("Urban Mean minus Rural Mean in Percentage", fontsize=11)
ax.set_title("Urban Advantage and Disadvantage - Blue is Urban Better, Red is Rural Better",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, diff_vals):
    ax.text(val + (0.3 if val > 0 else -0.3), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center",
            ha="left" if val > 0 else "right", fontsize=8)
plt.tight_layout()
plt.show()
plt.close()

# Final Summary Dashboard

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("INDIA DISTRICT HEALTH SURVEY NFHS 2015\nExecutive Summary Dashboard",
             fontsize=18, fontweight="bold", color="white", y=0.98)

gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

kpi_data = [
    ("Avg Women Literacy", f"{df_total['Women_Literacy'].mean():.1f}%", "#2ecc71"),
    ("Avg Child Stunting", f"{df_total['Stunting'].mean():.1f}%", "#e74c3c"),
    ("Avg Child Anaemia", f"{df_total['Child_Anaemia'].mean():.1f}%", "#f39c12"),
    ("Institutional Births", f"{df_total['Institutional_Births'].mean():.1f}%", "#3498db"),
    ("Child Marriage Rate", f"{df_total['Child_Marriage_Women'].mean():.1f}%", "#9b59b6"),
    ("Full Immunization", f"{df_total['Full_Immunization'].mean():.1f}%", "#1abc9c"),
    ("Clean Cooking Fuel", f"{df_total['Clean_Cooking_Fuel'].mean():.1f}%", "#e67e22"),
    ("Improved Sanitation", f"{df_total['Improved_Sanitation'].mean():.1f}%", "#34495e"),
]

for idx, (title, value, color) in enumerate(kpi_data):
    r, c = divmod(idx, 4)
    ax_kpi = fig.add_subplot(gs[r, c])
    ax_kpi.set_facecolor(color)
    ax_kpi.text(0.5, 0.60, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color="white", transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.25, title, ha="center", va="center",
                fontsize=9, color="white", style="italic", transform=ax_kpi.transAxes)
    ax_kpi.set_xticks([])
    ax_kpi.set_yticks([])
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(2)

plt.show()
plt.close()
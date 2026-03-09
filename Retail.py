"""
Consumer360 Retail Dashboard Analysis
======================================
Covers:
  1. Sales & Revenue Trends
  2. Customer Segmentation
  3. Churn Risk Analysis
  4. Product & Category Performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "C:/Users/gagan/Downloads/Consumer360_Retail_Dashboard_Dataset.csv"  # update path if needed
OUTPUT_DIR = "."                                          # folder for saved PNGs

PALETTE   = ["#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3", "#4895EF"]
BG_COLOR  = "#F8F9FA"
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
                     "font.family": "DejaVu Sans"})

# ── Load & Prep ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
df["Month"]         = df["Purchase_Date"].dt.to_period("M")
df["Month_dt"]      = df["Purchase_Date"].dt.to_period("M").dt.to_timestamp()
df["Quarter"]       = df["Purchase_Date"].dt.to_period("Q").astype(str)
df["Age_Group"]     = pd.cut(df["Customer_Age"],
                              bins=[0,25,35,45,55,100],
                              labels=["18-25","26-35","36-45","46-55","55+"])

print("✅  Dataset loaded:", df.shape[0], "rows ×", df.shape[1], "columns")
print("   Date range:", df["Purchase_Date"].min().date(), "→",
      df["Purchase_Date"].max().date())


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SALES & REVENUE TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sales_revenue():
    monthly = (df.groupby("Month_dt")
                 .agg(Revenue=("Total_Amount","sum"),
                      Orders=("Transaction_ID","count"),
                      AOV=("Total_Amount","mean"))
                 .reset_index())

    quarterly = (df.groupby("Quarter")
                   .agg(Revenue=("Total_Amount","sum"))
                   .reset_index())

    channel_rev = (df.groupby("Channel")["Total_Amount"]
                     .sum().sort_values(ascending=False))

    city_rev = (df.groupby("City")["Total_Amount"]
                  .sum().sort_values(ascending=False))

    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle("Sales & Revenue Trends", fontsize=20, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Monthly Revenue
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(monthly["Month_dt"], monthly["Revenue"]/1e6,
                     alpha=0.18, color=PALETTE[0])
    ax1.plot(monthly["Month_dt"], monthly["Revenue"]/1e6,
             color=PALETTE[0], linewidth=2.5, marker="o", markersize=4)
    ax1.set_title("Monthly Revenue (₹ M)", fontweight="bold")
    ax1.set_xlabel(""); ax1.set_ylabel("Revenue (₹ Million)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, ha="right")

    # Monthly Orders
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(monthly["Month_dt"], monthly["Orders"],
            color=PALETTE[2], width=20, alpha=0.85)
    ax2.set_title("Monthly Order Volume", fontweight="bold")
    ax2.set_xlabel(""); ax2.set_ylabel("Orders")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")

    # Quarterly Revenue
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(quarterly["Quarter"], quarterly["Revenue"]/1e6,
                   color=PALETTE[:len(quarterly)], alpha=0.88)
    ax3.bar_label(bars, fmt="%.1fM", padding=3, fontsize=8)
    ax3.set_title("Quarterly Revenue (₹ M)", fontweight="bold")
    ax3.set_xlabel(""); ax3.set_ylabel("Revenue (₹ Million)")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25, ha="right")

    # Revenue by Channel
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.barh(channel_rev.index, channel_rev.values/1e6,
                    color=PALETTE[:len(channel_rev)], alpha=0.88)
    ax4.bar_label(bars, fmt="%.1fM", padding=3, fontsize=8)
    ax4.set_title("Revenue by Channel (₹ M)", fontweight="bold")
    ax4.set_xlabel("Revenue (₹ Million)")

    # Revenue by City
    ax5 = fig.add_subplot(gs[1, 2])
    bars = ax5.barh(city_rev.index, city_rev.values/1e6,
                    color=PALETTE[:len(city_rev)], alpha=0.88)
    ax5.bar_label(bars, fmt="%.1fM", padding=3, fontsize=8)
    ax5.set_title("Revenue by City (₹ M)", fontweight="bold")
    ax5.set_xlabel("Revenue (₹ Million)")

    plt.savefig(f"{OUTPUT_DIR}/1_sales_revenue_trends.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Chart saved: 1_sales_revenue_trends.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_customer_segmentation():
    age_spend  = df.groupby("Age_Group", observed=True)["Total_Amount"].mean()
    gender_rev = df.groupby("Customer_Gender")["Total_Amount"].sum()
    income_seg = (df.groupby(["Income_Level","Channel"], observed=True)
                    ["Total_Amount"].sum().unstack(fill_value=0))
    payment_dist = df["Payment_Mode"].value_counts()
    rating_seg   = df.groupby("Age_Group", observed=True)["Customer_Rating"].mean()
    coupon_impact = (df.groupby("Coupon_Used")
                       .agg(AvgSpend=("Total_Amount","mean"),
                            Count=("Transaction_ID","count"))
                       .reset_index())

    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle("Customer Segmentation", fontsize=20, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Avg spend by Age Group
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(age_spend.index.astype(str), age_spend.values,
                   color=PALETTE, alpha=0.88)
    ax1.bar_label(bars, fmt="₹%.0f", padding=3, fontsize=8)
    ax1.set_title("Avg Spend by Age Group", fontweight="bold")
    ax1.set_xlabel("Age Group"); ax1.set_ylabel("Avg Transaction (₹)")

    # Revenue by Gender (pie)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(gender_rev, labels=gender_rev.index, autopct="%1.1f%%",
            colors=PALETTE[:2], startangle=140, wedgeprops=dict(edgecolor="white"))
    ax2.set_title("Revenue Split by Gender", fontweight="bold")

    # Income × Channel stacked bar
    ax3 = fig.add_subplot(gs[0, 2])
    income_seg.plot(kind="bar", ax=ax3, color=PALETTE[:3], alpha=0.88,
                    edgecolor="white")
    ax3.set_title("Revenue: Income × Channel", fontweight="bold")
    ax3.set_xlabel("Income Level"); ax3.set_ylabel("Revenue (₹)")
    ax3.legend(title="Channel", fontsize=8)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

    # Payment mode distribution
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(payment_dist.index, payment_dist.values,
                   color=PALETTE[:4], alpha=0.88)
    ax4.bar_label(bars, padding=3, fontsize=8)
    ax4.set_title("Payment Mode Distribution", fontweight="bold")
    ax4.set_xlabel(""); ax4.set_ylabel("Transactions")

    # Avg rating by Age Group
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(rating_seg.index.astype(str), rating_seg.values,
             color=PALETTE[1], marker="D", linewidth=2.5, markersize=8)
    ax5.set_ylim(1, 5)
    ax5.set_title("Avg Customer Rating by Age Group", fontweight="bold")
    ax5.set_xlabel("Age Group"); ax5.set_ylabel("Avg Rating (1-5)")
    ax5.axhline(rating_seg.mean(), linestyle="--", color="gray",
                linewidth=1, label=f"Overall avg: {rating_seg.mean():.2f}")
    ax5.legend(fontsize=8)

    # Coupon impact on spend
    ax6 = fig.add_subplot(gs[1, 2])
    bars = ax6.bar(coupon_impact["Coupon_Used"], coupon_impact["AvgSpend"],
                   color=[PALETTE[3], PALETTE[0]], alpha=0.88)
    ax6.bar_label(bars, fmt="₹%.0f", padding=3, fontsize=9)
    ax6.set_title("Coupon Usage vs Avg Spend", fontweight="bold")
    ax6.set_xlabel("Coupon Used"); ax6.set_ylabel("Avg Spend (₹)")

    plt.savefig(f"{OUTPUT_DIR}/2_customer_segmentation.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Chart saved: 2_customer_segmentation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CHURN RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_churn_risk():
    churn_dist   = df["Churn_Risk"].value_counts()
    churn_city   = (df.groupby(["City","Churn_Risk"])
                      ["Customer_ID"].count().unstack(fill_value=0))
    churn_tenure = (df.groupby(["Churn_Risk","Age_Group"], observed=True)
                      ["Customer_Tenure_Months"].mean().unstack(fill_value=0))
    churn_rating = df.groupby("Churn_Risk")["Customer_Rating"].mean()
    churn_rev    = df.groupby("Churn_Risk")["Total_Amount"].mean()
    churn_loyalty= df.groupby("Churn_Risk")["Loyalty_Points_Earned"].mean()

    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle("Churn Risk Analysis", fontsize=20, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Churn Risk Distribution (donut)
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(
        churn_dist, labels=churn_dist.index, autopct="%1.1f%%",
        colors=[PALETTE[2], PALETTE[0], PALETTE[1]],
        startangle=90, wedgeprops=dict(edgecolor="white", width=0.6))
    ax1.set_title("Churn Risk Distribution", fontweight="bold")

    # Churn by City (stacked bar)
    ax2 = fig.add_subplot(gs[0, 1:])
    churn_city.plot(kind="bar", ax=ax2, stacked=True,
                    color=[PALETTE[2], PALETTE[0], PALETTE[1]],
                    alpha=0.88, edgecolor="white")
    ax2.set_title("Churn Risk by City", fontweight="bold")
    ax2.set_xlabel(""); ax2.set_ylabel("Customer Count")
    ax2.legend(title="Churn Risk", fontsize=8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha="right")

    # Avg Rating by Churn Risk
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(churn_rating.index, churn_rating.values,
                   color=[PALETTE[2], PALETTE[0], PALETTE[1]], alpha=0.88)
    ax3.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax3.set_ylim(0, 5.5)
    ax3.set_title("Avg Rating by Churn Risk", fontweight="bold")
    ax3.set_xlabel("Churn Risk"); ax3.set_ylabel("Avg Rating")

    # Avg Revenue by Churn Risk
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(churn_rev.index, churn_rev.values,
                   color=[PALETTE[2], PALETTE[0], PALETTE[1]], alpha=0.88)
    ax4.bar_label(bars, fmt="₹%.0f", padding=3, fontsize=9)
    ax4.set_title("Avg Spend by Churn Risk", fontweight="bold")
    ax4.set_xlabel("Churn Risk"); ax4.set_ylabel("Avg Spend (₹)")

    # Avg Loyalty Points by Churn Risk
    ax5 = fig.add_subplot(gs[1, 2])
    bars = ax5.bar(churn_loyalty.index, churn_loyalty.values,
                   color=[PALETTE[2], PALETTE[0], PALETTE[1]], alpha=0.88)
    ax5.bar_label(bars, fmt="%.0f pts", padding=3, fontsize=9)
    ax5.set_title("Avg Loyalty Points by Churn Risk", fontweight="bold")
    ax5.set_xlabel("Churn Risk"); ax5.set_ylabel("Avg Loyalty Points")

    plt.savefig(f"{OUTPUT_DIR}/3_churn_risk_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Chart saved: 3_churn_risk_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PRODUCT & CATEGORY PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_product_category():
    cat_rev      = df.groupby("Product_Category")["Total_Amount"].sum().sort_values()
    cat_qty      = df.groupby("Product_Category")["Quantity"].sum().sort_values()
    brand_rev    = df.groupby("Brand")["Total_Amount"].sum().sort_values()
    cat_return   = (df.groupby("Product_Category")["Return_Status"]
                      .apply(lambda x: (x == "Returned").mean() * 100).sort_values())
    cat_rating   = df.groupby("Product_Category")["Customer_Rating"].mean().sort_values()
    cat_discount = df.groupby("Product_Category")["Discount_%"].mean().sort_values()

    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle("Product & Category Performance", fontsize=20,
                 fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Revenue by Category
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(cat_rev.index, cat_rev.values/1e6,
                    color=PALETTE[:len(cat_rev)], alpha=0.88)
    ax1.bar_label(bars, fmt="%.1fM", padding=3, fontsize=8)
    ax1.set_title("Revenue by Category (₹ M)", fontweight="bold")
    ax1.set_xlabel("Revenue (₹ Million)")

    # Units Sold by Category
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(cat_qty.index, cat_qty.values,
                    color=PALETTE[:len(cat_qty)], alpha=0.88)
    ax2.bar_label(bars, padding=3, fontsize=8)
    ax2.set_title("Units Sold by Category", fontweight="bold")
    ax2.set_xlabel("Quantity Sold")

    # Revenue by Brand
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.barh(brand_rev.index, brand_rev.values/1e6,
                    color=PALETTE[:len(brand_rev)], alpha=0.88)
    ax3.bar_label(bars, fmt="%.1fM", padding=3, fontsize=8)
    ax3.set_title("Revenue by Brand (₹ M)", fontweight="bold")
    ax3.set_xlabel("Revenue (₹ Million)")

    # Return Rate by Category
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.barh(cat_return.index, cat_return.values,
                    color=PALETTE[:len(cat_return)], alpha=0.88)
    ax4.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax4.set_title("Return Rate by Category (%)", fontweight="bold")
    ax4.set_xlabel("Return Rate (%)")

    # Avg Rating by Category
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.barh(cat_rating.index, cat_rating.values,
                    color=PALETTE[:len(cat_rating)], alpha=0.88)
    ax5.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax5.set_xlim(0, 5.5)
    ax5.set_title("Avg Customer Rating by Category", fontweight="bold")
    ax5.set_xlabel("Avg Rating (1–5)")

    # Avg Discount by Category
    ax6 = fig.add_subplot(gs[1, 2])
    bars = ax6.barh(cat_discount.index, cat_discount.values,
                    color=PALETTE[:len(cat_discount)], alpha=0.88)
    ax6.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax6.set_title("Avg Discount by Category (%)", fontweight="bold")
    ax6.set_xlabel("Avg Discount (%)")

    plt.savefig(f"{OUTPUT_DIR}/4_product_category_performance.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅  Chart saved: 4_product_category_performance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY STATS  (printed to console)
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "="*55)
    print("  CONSUMER360 — KEY METRICS SUMMARY")
    print("="*55)
    print(f"  Total Revenue       : ₹{df['Total_Amount'].sum():,.0f}")
    print(f"  Total Transactions  : {len(df):,}")
    print(f"  Unique Customers    : {df['Customer_ID'].nunique():,}")
    print(f"  Avg Order Value     : ₹{df['Total_Amount'].mean():,.0f}")
    print(f"  Avg Customer Rating : {df['Customer_Rating'].mean():.2f} / 5")
    print(f"  Overall Return Rate : {(df['Return_Status']=='Returned').mean()*100:.1f}%")
    top_cat  = df.groupby('Product_Category')['Total_Amount'].sum().idxmax()
    top_city = df.groupby('City')['Total_Amount'].sum().idxmax()
    top_brand= df.groupby('Brand')['Total_Amount'].sum().idxmax()
    print(f"  Top Category        : {top_cat}")
    print(f"  Top City            : {top_city}")
    print(f"  Top Brand           : {top_brand}")
    high_churn_pct = (df['Churn_Risk']=='High').mean()*100
    print(f"  High Churn Risk     : {high_churn_pct:.1f}% of customers")
    print("="*55 + "\n")


# ── Run everything ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_summary()
    plot_sales_revenue()
    plot_customer_segmentation()
    plot_churn_risk()
    plot_product_category()
    print("\n🎉  All 4 analysis charts generated successfully!")
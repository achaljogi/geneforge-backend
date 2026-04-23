from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import json
from io import BytesIO

app = FastAPI(title="Gene Expression Analysis API")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running successfully"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= Upload CSV =================
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):

    contents = await file.read()

    try:
        df = pd.read_csv(BytesIO(contents), engine="c")
        preview = df.head(20)

        return {
            "columns": df.columns.tolist(),
            "data": preview.to_dict(orient="records"),
            "rows": len(df)
        }

    except Exception as e:
        return {"error": str(e)}

# ================= Differential Expression =================
@app.post("/differential-expression")
async def differential_expression(
    file: UploadFile = File(...),
    control_samples: str = Form(...),
    treated_samples: str = Form(...)
):

    df = pd.read_csv(file.file, engine="c")

    gene_col = df.columns[0]

    control_cols = json.loads(control_samples)
    treat_cols = json.loads(treated_samples)

    control_data = df[control_cols].astype(float)
    treat_data = df[treat_cols].astype(float)

    # ===== Mean expression =====
    control_mean = control_data.mean(axis=1)
    treat_mean = treat_data.mean(axis=1)

    # ===== Log2 Fold Change =====
    log2fc = np.log2((treat_mean + 1e-6) / (control_mean + 1e-6))

    # ===== T-Test =====
    pvalues = ttest_ind(
        control_data,
        treat_data,
        axis=1,
        equal_var=False
    ).pvalue

    # ===== FDR correction =====
    adj_pvalues = multipletests(pvalues, method="fdr_bh")[1]

    # ===== Volcano Plot Y Axis =====
    neg_log10_pvalue = -np.log10(pvalues + 1e-10)

    result_df = pd.DataFrame({
        "Gene": df[gene_col],
        "ControlMean": control_mean.round(4),
        "TreatmentMean": treat_mean.round(4),
        "log2FoldChange": log2fc.round(4),
        "pvalue": pvalues,
        "adj_pvalue": adj_pvalues,
        "neg_log10_pvalue": neg_log10_pvalue.round(4)
    })

    return result_df.to_dict(orient="records")


# ================= Heatmap =================
@app.post("/top-genes-heatmap")
async def top_genes_heatmap(
    file: UploadFile = File(...),
    control_samples: str = Form(...),
    treated_samples: str = Form(...),
    top_genes: int = Form(20)
):

    df = pd.read_csv(file.file, engine="c")

    gene_col = df.columns[0]

    control_cols = json.loads(control_samples)
    treat_cols = json.loads(treated_samples)

    selected_cols = [gene_col] + control_cols + treat_cols
    df = df[selected_cols]

    control_data = df[control_cols].astype(float)
    treat_data = df[treat_cols].astype(float)

    # Mean expression
    df["ControlMean"] = control_data.mean(axis=1)
    df["TreatMean"] = treat_data.mean(axis=1)

    # Log2 Fold Change
    df["log2FC"] = np.log2((df["TreatMean"] + 1e-6) / (df["ControlMean"] + 1e-6))

    # Sort by absolute fold change
    df = df.reindex(df["log2FC"].abs().sort_values(ascending=False).index)

    # Select top genes
    top = df.head(int(top_genes))

    heatmap_df = top[[gene_col] + control_cols + treat_cols]

    return heatmap_df.to_dict(orient="records")
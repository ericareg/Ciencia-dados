import joblib, pandas as pd, numpy as np
art = joblib.load("xgb_model.joblib")
pre, model = art["preprocess"], art["model"]

# nomes das features (numéricas + OneHot)
names = []
for name, trans, cols in pre.transformers_:
    if name == "num":
        names += list(cols)
    elif name == "cat":
        ohe = trans.named_steps["onehot"]
        names += list(ohe.get_feature_names_out(cols))

imp = model.feature_importances_
df_imp = (pd.DataFrame({"feature": names[:len(imp)], "importance": imp[:len(names)]})
            .sort_values("importance", ascending=False))
df_imp.to_csv("feature_importance.csv", index=False)

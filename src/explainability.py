import shap
import pandas as pd

class FraudExplainer:
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def get_top_risk_drivers(self, feature_df: pd.DataFrame, top_n: int = 3) -> list:
        shap_values = self.explainer(feature_df)
        sample_shap = shap_values[0].values
        
        contributions = pd.DataFrame({
            'feature': feature_df.columns,
            'value': feature_df.iloc[0].values,
            'shap_impact': sample_shap
        }).sort_values(by='shap_impact', ascending=False)
        
        return contributions.head(top_n).to_dict(orient='records')
import os
from openai import OpenAI

class AMLReportAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

    def _get_active_model(self) -> str:
        """Dynamically retrieve the first available chat model from Groq."""
        try:
            models_list = self.client.models.list()
            text_models = [
                m.id for m in models_list.data 
                if not any(tag in m.id.lower() for tag in ['whisper', 'guard', 'tts', 'audio'])
            ]
            # Prioritize standard Groq models if available
            for preferred in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                if preferred in text_models:
                    return preferred
            return text_models[0] if text_models else "llama-3.3-70b-versatile"
        except Exception:
            return "llama-3.3-70b-versatile"

    def generate_report(self, prob: float, drivers: list) -> str:
        active_model = self._get_active_model()

        prompt = f"""
[SYSTEM INSTRUCTION]
You are a Lead Financial Crime Compliance Analyst. Write a 3-sentence Risk Report for the AML team.

[TRANSACTION RISK DATA]
Calculated Fraud Probability: {prob:.2%}
Top Risk Feature Drivers:
1. Feature: {drivers[0]['feature']} | Observed: {drivers[0]['value']} | Score: +{drivers[0]['shap_impact']:.3f}
2. Feature: {drivers[1]['feature']} | Observed: {drivers[1]['value']} | Score: +{drivers[1]['shap_impact']:.3f}
3. Feature: {drivers[2]['feature']} | Observed: {drivers[2]['value']} | Score: +{drivers[2]['shap_impact']:.3f}

[REQUIRED OUTPUT FORMAT]
- Summary Statement: Primary reason for flag.
- Evidence Breakdown: Concrete behavior captured by top 2 SHAP drivers.
- Recommended Compliance Action: Freeze account, request identity re-verification, or clear flag.
"""
        try:
            response = self.client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": "You are a Lead Financial Crime Compliance Analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq API Error ({active_model}): {str(e)}"
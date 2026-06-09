import pandas as pd

class RiskLabeler:
    def __init__(self):
        self.w_vol = 0.4
        self.w_beta = 0.4
        self.w_sector = 0.2

        # === تم تضييق الحدود لتناسب السوق السعودي المستقر وتسمح بظهور High Risk ===
        self.min_port_vol, self.max_port_vol = 0.05, 0.30     # أي تذبذب فوق 30% يعتبر خطير جداً
        self.min_port_beta, self.max_port_beta = 0.5, 1.3     # أي بيتا فوق 1.3 تعتبر خطيرة جداً
        self.min_sec_vol, self.max_sec_vol = 0.05, 0.30      
        self.min_sec_beta, self.max_sec_beta = 0.5, 1.3      
        self.min_sec_risk, self.max_sec_risk = 0.0, 100.0    

    def min_max_normalization(self, value, min_val, max_val):
        if value < min_val: value = min_val
        if value > max_val: value = max_val
        return ((value - min_val) / (max_val - min_val)) * 100.0

    def step_3_sector_risk_score(self, sector_vol, sector_beta):
        score_vol = self.min_max_normalization(sector_vol, self.min_sec_vol, self.max_sec_vol)
        score_beta = self.min_max_normalization(sector_beta, self.min_sec_beta, self.max_sec_beta)
        sector_risk = (0.5 * score_vol) + (0.5 * score_beta)
        return sector_risk

    def step_4_normalize_risk_metrics(self, port_vol, port_beta, sector_risk):
        norm_vol = self.min_max_normalization(port_vol, self.min_port_vol, self.max_port_vol)
        norm_beta = self.min_max_normalization(port_beta, self.min_port_beta, self.max_port_beta)
        norm_sector = self.min_max_normalization(sector_risk, self.min_sec_risk, self.max_sec_risk)
        return norm_vol, norm_beta, norm_sector

    def step_5_compute_risk_score(self, norm_vol, norm_beta, norm_sector):
        risk_score = (self.w_vol * norm_vol) + (self.w_beta * norm_beta) + (self.w_sector * norm_sector)
        return risk_score

    def step_6_label_data(self, risk_score):
        if risk_score <= 33.0: return "Low Risk"
        elif risk_score <= 66.0: return "Medium Risk"
        else: return "High Risk"

    def calculate_final_score(self, port_q_pct, port_b, sector_q, sector_b):
        port_vol_decimal = port_q_pct / 100.0
        raw_sector_risk = self.step_3_sector_risk_score(sector_q, sector_b)
        norm_vol, norm_beta, norm_sector = self.step_4_normalize_risk_metrics(port_vol_decimal, port_b, raw_sector_risk)
        final_score = self.step_5_compute_risk_score(norm_vol, norm_beta, norm_sector)
        label = self.step_6_label_data(final_score)

        return {
            "Final_Risk_Score": round(final_score, 2),
            "Risk_Category": label,
            "Details": {
                "Normalized_Port_Vol": round(norm_vol, 2),
                "Normalized_Port_Beta": round(norm_beta, 2),
                "Raw_Sector_Risk": round(raw_sector_risk, 2),
                "Normalized_Sector_Risk": round(norm_sector, 2)
            }
        }
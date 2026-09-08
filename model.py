import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import os


class MOILManganeseAI:
    # Known Indian manganese-ore belts the app's dropdown can switch between.
    # lat/lon ranges only decide where points scatter on the map — the
    # trained grade model doesn't use location as a feature, so any belt's
    # points can be scored with the same model.
    BELTS = {
        "Balaghat-Nagpur (MP / Maharashtra)": {
            "lat_range": (21.4, 21.9), "lon_range": (79.5, 80.5),
            "center": {"lat": 21.65, "lon": 80.0}, "zoom": 7,
        },
        "Keonjhar-Sundargarh (Odisha)": {
            "lat_range": (21.6, 22.3), "lon_range": (85.2, 85.9),
            "center": {"lat": 21.95, "lon": 85.55}, "zoom": 7,
        },
        "Bellary-Hospet (Karnataka)": {
            "lat_range": (15.0, 15.4), "lon_range": (76.3, 76.9),
            "center": {"lat": 15.2, "lon": 76.6}, "zoom": 8,
        },
        "North Goa Belt": {
            "lat_range": (15.35, 15.65), "lon_range": (73.9, 74.2),
            "center": {"lat": 15.5, "lon": 74.05}, "zoom": 9,
        },
    }

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.reserve_model = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state)
        self.shortfall_model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state)
        self.is_trained = False

    def generate_belt_data(self, belt_name, n_samples=250):
        """
        Generates simulated survey points (lat/lon + satellite/geological
        features) scattered within the chosen belt's bounding box, ready to
        be scored by predict_reserve_grid(). Does not include mn_grade_pct —
        that's the model's job.
        """
        cfg = self.BELTS[belt_name]
        rng = np.random.default_rng(self.random_state)

        lats = rng.uniform(*cfg["lat_range"], n_samples)
        lons = rng.uniform(*cfg["lon_range"], n_samples)
        depth = rng.uniform(15.0, 220.0, n_samples)
        ndvi = rng.uniform(0.12, 0.65, n_samples)
        lst_c = rng.uniform(28.0, 44.0, n_samples)
        soil_moisture = rng.uniform(12.0, 48.0, n_samples)
        magnetic_susceptibility = rng.uniform(10.0, 95.0, n_samples)

        return pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'depth_m': depth,
            'ndvi': ndvi,
            'lst_celsius': lst_c,
            'soil_moisture_pct': soil_moisture,
            'mag_susceptibility': magnetic_susceptibility,
        })

    def generate_space_reserve_data(self, n_samples=600):
        """
        Simulates geospatial drilling data fused with Space Technology / Earth Observation inputs:
        - Lat/Lon bounded around Central India manganese belt (Balaghat, MP & Nagpur/Bhandara, MH)
        - Surface Spectral Indicators: NDVI (Vegetation Index), LST (Land Surface Temperature °C), Soil Moisture (%)
        - Sub-surface indicators: Depth (m), Geological host rock conductivity index
        - Target: Mn Grade (%)
        """
        np.random.seed(self.random_state)

        lats = np.random.uniform(21.4, 21.9, n_samples)
        lons = np.random.uniform(79.5, 80.5, n_samples)
        depth = np.random.uniform(15.0, 220.0, n_samples)

        # Satellite indicators
        ndvi = np.random.uniform(0.12, 0.65, n_samples)
        lst_c = np.random.uniform(28.0, 44.0, n_samples)
        soil_moisture = np.random.uniform(12.0, 48.0, n_samples)
        magnetic_susceptibility = np.random.uniform(10.0, 95.0, n_samples)

        # Subsurface grade generation correlating with mineralization indicators
        mn_grade = (
            22.0
            + 14.0 * (1.0 - ndvi)
            + 0.12 * lst_c
            + 0.08 * magnetic_susceptibility
            - 0.03 * depth
            + np.random.normal(0, 3.5, n_samples)
        )
        mn_grade = np.clip(mn_grade, 12.0, 52.0)

        df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'depth_m': depth,
            'ndvi': ndvi,
            'lst_celsius': lst_c,
            'soil_moisture_pct': soil_moisture,
            'mag_susceptibility': magnetic_susceptibility,
            'mn_grade_pct': mn_grade
        })
        return df

    def generate_operational_telemetry(self, n_days=365):
        """
        Simulates mine-level daily operational logs.
        Target: Shortfall occurrence (1 = Shortfall, 0 = Target Met)
        """
        np.random.seed(self.random_state)
        dates = pd.date_range(start="2023-04-01", periods=n_days, freq="D")

        # Monsoon pattern simulation (July-Sept heavy rain)
        day_of_year = dates.dayofyear
        is_monsoon = (day_of_year >= 165) & (day_of_year <= 265)

        rainfall = np.where(is_monsoon, np.random.gamma(
            shape=2.5, scale=12.0, size=n_days), np.random.exponential(scale=2.0, size=n_days))
        downtime = np.random.gamma(shape=2.0, scale=1.8, size=n_days)
        blasting_delay = np.where(rainfall > 25.0, np.random.uniform(
            2.5, 7.0, n_days), np.random.uniform(0.0, 1.8, n_days))
        wagon_avail = np.clip(np.random.normal(0.92, 0.10, n_days), 0.5, 1.0)

        shortfall_risk_score = (
            0.35 * (rainfall / 30.0)
            + 0.35 * (downtime / 6.0)
            + 0.20 * (blasting_delay / 4.0)
            + 0.10 * (1.0 - wagon_avail)
        )
        is_shortfall = (shortfall_risk_score > 0.65).astype(int)

        df = pd.DataFrame({
            'date': dates,
            'rainfall_mm': np.round(rainfall, 2),
            'equipment_downtime_hrs': np.round(downtime, 2),
            'blasting_delay_hrs': np.round(blasting_delay, 2),
            'wagon_availability_ratio': np.round(wagon_avail, 2),
            'is_shortfall': is_shortfall
        })
        return df

    def train_models(self):
        """Trains both the reserve estimation and shortfall prediction models."""
        # 1. Train Reserve Grade Regressor
        reserve_df = self.generate_space_reserve_data()
        X_res = reserve_df[['latitude', 'longitude', 'depth_m', 'ndvi',
                            'lst_celsius', 'soil_moisture_pct', 'mag_susceptibility']]
        y_res = reserve_df['mn_grade_pct']
        self.reserve_model.fit(X_res, y_res)

        # 2. Train Shortfall Classifier
        ops_df = self.generate_operational_telemetry()
        X_ops = ops_df[['rainfall_mm', 'equipment_downtime_hrs',
                        'blasting_delay_hrs', 'wagon_availability_ratio']]
        y_ops = ops_df['is_shortfall']
        self.shortfall_model.fit(X_ops, y_ops)

        self.is_trained = True
        return reserve_df, ops_df

    def predict_reserve_grid(self, input_features_df):
        """Predicts Mn Grade % over arbitrary survey or grid points."""
        return self.reserve_model.predict(input_features_df)

    def predict_shortfall_risk(self, rainfall, downtime, blasting_delay, wagon_avail):
        """Predicts probability of a production shortfall on a given day."""
        X_in = pd.DataFrame([{
            'rainfall_mm': rainfall,
            'equipment_downtime_hrs': downtime,
            'blasting_delay_hrs': blasting_delay,
            'wagon_availability_ratio': wagon_avail
        }])
        prob_shortfall = self.shortfall_model.predict_proba(X_in)[0][1]
        pred_label = self.shortfall_model.predict(X_in)[0]
        return prob_shortfall, pred_label

    @staticmethod
    def get_prescriptive_actions(rainfall, downtime, blasting_delay, wagon_avail):
        """Generates prioritized corrective actions based on operational bottlenecks."""
        actions = []
        if rainfall > 25.0:
            actions.append(
                "[Weather Action]: Heavy precipitation detected (>25mm). Divert haul trucks to high-bench all-weather haul roads and activate sump dewatering pumps.")
        if downtime > 4.5:
            actions.append(
                "[Equipment Action]: Critical equipment downtime (>4.5 hrs). Re-route secondary excavators from waste overburden stripping to active high-grade ore face.")
        if blasting_delay > 2.0:
            actions.append(
                "[Blasting Action]: Blasting delay exceeding 2 hrs. Shift blast window to secondary evening shift; feed crushing plant from buffer mine-head stock.")
        if wagon_avail < 0.80:
            actions.append(
                "[Logistics Action]: Rail rake availability deficit (<80%). Redirect secondary production dispatches to road transport to prevent mine-head stock accumulation.")
        if not actions:
            actions.append(
                "[Nominal Operations]: All parameters within safe operating window. Continue standard extraction and haul cycle.")
        return actions

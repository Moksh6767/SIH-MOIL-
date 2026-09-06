# MOIL Manganese AI Reserve Forecaster

This repository contains an end-to-end AI/ML and Space Technology pipeline developed to solve MOIL Limited's challenge of identifying manganese reserves and mitigating production shortfalls.

## Architecture

1. **Space Tech Integration**: Fuses synthetic Sentinel-2/Bhuvan satellite data (NDVI, LST, Soil Moisture).
2. **AI/ML Models**:
   - Spatial Random Forest for reserve grade prediction.
   - Time-series classifier for predicting production bottlenecks (weather, downtime).
3. **Prescriptive Engine**: Recommends dynamic equipment redeployment.
4. **Dashboard**: Interactive Streamlit UI for mine managers.

## Setup

```bash
pip install -r requirements.txt
# OR using conda
conda env create -f environment.yml
conda activate moil-ai
```

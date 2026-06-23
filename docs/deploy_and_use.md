# Deploy and Use

You can use the engine in two ways:

1. Locally from the command line.
2. As a hosted API service.

## Local Use

Create sample data:

```powershell
python -m football_prediction_engine.cli sample --rows 240 --output outputs/sample_matches.csv
```

Run a prediction:

```powershell
python -m football_prediction_engine.cli predict --input outputs/sample_matches.csv --home Arsenal --away "Manchester City"
```

Run professional validation:

```powershell
python -m football_prediction_engine.cli validate --input outputs/sample_matches.csv --full-report
```

## Run API Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create sample data:

```powershell
python -m football_prediction_engine.cli sample --rows 240 --output outputs/sample_matches.csv
```

Start the API:

```powershell
uvicorn football_prediction_engine.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## API Examples

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Predict a match:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"home_team":"Arsenal","away_team":"Manchester City","odds_home_implied":0.42,"odds_draw_implied":0.27,"odds_away_implied":0.31}'
```

Run validation:

```powershell
Invoke-RestMethod http://localhost:8000/validate
```

## Railway Deployment

1. Push this folder to GitHub.
2. Create a new Railway project.
3. Choose "Deploy from GitHub repo".
4. Railway will use the `Procfile`.
5. After deployment, open the generated Railway URL.
6. Visit `/docs` to use the API interactively.

The API creates sample data automatically on first prediction or validation if `outputs/sample_matches.csv` is missing. For real production use, replace `outputs/sample_matches.csv` with a mounted database, object storage, or real ETL output.

## Production Notes

- Do not treat the sample data as real betting advice.
- Add real data ingestion before relying on predictions.
- Keep closing odds as the benchmark.
- Use `/validate` before trusting any new model or data source.
- Skip matches with weak edge, poor data quality, or high model disagreement.

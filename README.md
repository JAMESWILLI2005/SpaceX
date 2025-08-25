Falcon 9 Landing Predictor: ML Models for Reuse Economics 🚀

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/JAMESWILLI2005/SpaceX/releases)

[![decision-tree](https://img.shields.io/badge/decision--tree-DecisionTree-green)](https://github.com/topics/decision-tree)
[![knn](https://img.shields.io/badge/knn-KNN-yellowgreen)](https://github.com/topics/knn)
[![logistic-regression](https://img.shields.io/badge/logistic--regression-LogReg-orange)](https://github.com/topics/logistic-regression)
[![prediction](https://img.shields.io/badge/prediction-Prediction-blueviolet)](https://github.com/topics/prediction)
[![preprocessing](https://img.shields.io/badge/preprocessing-Prep-cadetblue)](https://github.com/topics/preprocessing)
[![python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![sklearn](https://img.shields.io/badge/scikit--learn-sklearn-ff69b4)](https://scikit-learn.org/)
[![svm](https://img.shields.io/badge/svm-SVM-darkgreen)](https://github.com/topics/svm)
[![webscraping](https://img.shields.io/badge/webscraping-Scrape-skyblue)](https://github.com/topics/webscraping)

Cover image
![Falcon 9 liftoff](https://upload.wikimedia.org/wikipedia/commons/5/53/Falcon_9_liftoff.jpg)

Table of contents
- Project
- Key ideas
- Data sources
- Pipeline
- Models
- Metrics and validation
- How to run (local)
- Releases
- Project layout
- Contributing
- License

Project
This repository predicts whether a Falcon 9 first stage will land after stage separation. SpaceX sells Falcon 9 launches for about $62 million. Other providers charge more than $165 million. A big driver of that margin is reusability. If the first stage lands, SpaceX reuses hardware and cuts launch cost per mission. This project frames landing as a binary classification task and supplies a reproducible pipeline from data retrieval to model export.

Key ideas
- Treat landing as binary classification: landed vs not landed.
- Use telemetry features, flight profile, payload mass, weather, and booster history.
- Prepare data with deterministic preprocessing: impute, encode, scale.
- Try multiple classifiers: logistic regression, decision tree, KNN, SVM.
- Validate with stratified cross-validation and time-aware splits where appropriate.
- Export the best model for on-demand inference and batch scoring.

Data sources
- Public mission pages and press kits (web scraping).
- Launch manifests and payload specifications.
- Telemetry snippets from public telemetry dumps or simulated signals.
- Weather archives at launch time (wind, pressure, temperature).
- Booster history: prior landings, age, and refurbishment cycle.

Pipeline
1. Web scraping
   - Scrape mission pages for mission name, booster core serial, landing outcome.
   - Scrape payload mass and orbit type.
   - Collect launch date/time for weather lookup.
2. Data join
   - Link telemetry and manifest rows by mission ID.
   - Consolidate booster history via serial lookup.
3. Preprocessing
   - Impute numeric fields with median per class.
   - Encode categorical fields with target encoding or one-hot where cardinality is low.
   - Scale numeric features with StandardScaler or RobustScaler.
   - Create derived features: mass-to-thrust ratio, launch azimuth bucket, time-since-last-flight.
4. Feature selection
   - Use mutual information and recursive feature elimination on a validation fold.
5. Modeling
   - Train logistic regression, decision tree, KNN, and SVM.
   - Calibrate probabilities for logistic regression and SVM.
6. Evaluation
   - Use balanced accuracy, precision-recall, ROC AUC, and F1.
   - Produce confusion matrices and calibration curves.
7. Export
   - Save best model as a `joblib` artifact and a light infer script `predict.py`.

Models
- Logistic Regression
  - Interpretable coefficients, baseline probability.
  - Use L2 regularization and class weights for imbalance.
- Decision Tree
  - Fast, interpretable splits, may overfit without depth control.
- K-Nearest Neighbors (KNN)
  - Works well with local patterns; tune k and distance metric.
- Support Vector Machine (SVM)
  - Effective in high-dimension, use RBF kernel or linear kernel for speed.

Metrics and validation
- Use stratified K-fold (k=5) by landing label.
- For temporal data, prefer time-based split: train on older launches and validate on recent launches.
- Track:
  - Accuracy
  - Precision and recall for landed class
  - F1-score
  - ROC AUC
  - PR AUC
- Use SHAP values for model explainability on the deployed candidate.

How to run (local)
Requirements
- Python 3.8 or newer
- pip packages: pandas, numpy, scikit-learn, joblib, requests, beautifulsoup4, shap, matplotlib

Install
Run Python package install command:
`pip install -r requirements.txt`

Data
- Raw data comes from web scraping and public logs.
- The repository includes a small sample dataset for quick tests.
- For full training, run the scraper to rebuild the dataset from mission pages and weather archives.

Workflow (high level)
- Prepare environment.
- Run data collection or place CSVs in `data/raw`.
- Run preprocessing to create `data/processed`.
- Train models using `scripts/train_model.py`.
- Evaluate models using `scripts/evaluate.py`.
- Export the selected model to `models/`.

Usage examples
- To build features:
`python scripts/build_features.py --input data/raw --output data/processed`
- To train a model:
`python scripts/train_model.py --config configs/train_config.yaml`
- To run a single prediction:
`python scripts/predict.py --model models/best_model.joblib --input sample_input.json`

Releases
Download the packaged release artifact and execute the included prediction script. The releases page contains packaged assets such as `spacex_predictor_v1.zip`. After you download the release asset, extract and run `predict.py` or `run.sh` inside the package.

- Visit the releases and download the file:
https://github.com/JAMESWILLI2005/SpaceX/releases

If the release link does not work, check the "Releases" section on the repository page in the top bar of GitHub.

Release badge
[![Download Releases](https://img.shields.io/badge/Download-Releases-orange?logo=github)](https://github.com/JAMESWILLI2005/SpaceX/releases)

Project layout
- data/
  - raw/                # raw scraped files and source CSVs
  - processed/          # cleaned and feature-engineered data
- docs/                 # design notes, modeling decisions, diagrams
- models/               # trained model artifacts (joblib)
- notebooks/            # EDA and experiments (Jupyter)
- scripts/
  - build_features.py
  - train_model.py
  - evaluate.py
  - predict.py
- configs/              # config files for training and preprocessing
- requirements.txt
- README.md

Preprocessing details
- Outlier handling: cap numeric features at 1st and 99th percentiles.
- Missing values: impute with median (numeric) and mode (categorical) or use model-based imputer.
- Feature engineering highlights:
  - `mass_to_thrust = payload_mass / max_thrust`
  - `booster_experience = prior_landings`
  - `launch_hour_bucket = floor(launch_time.hour / 4)`
  - `sea_level_wind_bin` from observed wind speed

Hyperparameter recommendations
- Logistic Regression: C in [0.01, 0.1, 1, 10], solver `saga` for large data.
- Decision Tree: max_depth in [3, 5, 10, None], min_samples_leaf >= 5.
- KNN: n_neighbors in [3, 5, 7, 11], metric `minkowski`.
- SVM: C in [0.1, 1, 10], gamma `scale` or tuned for RBF.

Model explainability
- Use SHAP to explain predictions.
- Report global feature importance and local explanations for select flights.
- Visualize partial dependence for high-impact features such as `prior_landings`, `payload_mass`, and `wind_speed`.

Web scraping notes
- Respect robots.txt and public scraping rules.
- Throttle requests and cache HTML to avoid repeated hits.
- Store raw HTML and parse pages with BeautifulSoup.
- Extract mission ID, booster serial, landing outcome, payload mass, orbit, and launch time.

Example outputs
- Confusion matrix for the final model.
- ROC curve showing area under curve.
- SHAP summary plot of top 10 features.
- Calibration curve comparing predicted probability and observed frequency.

Continuous evaluation
- Re-run training when new missions enter the dataset.
- Monitor drift: check feature distribution and model score over time.
- Use a rolling window or incremental update if data grows.

Contributing
- Open an issue for bug reports and model requests.
- Create a fork and submit a pull request for code changes.
- Follow the code style in `scripts` and add unit tests for new functionality.
- Add reproducible experiments as notebooks under `notebooks/experiments`.

Code style and tests
- Use type hints and docstrings in public functions.
- Include unit tests for preprocessing and model I/O in `tests/`.
- Run tests with `pytest`.

Security and data privacy
- Do not commit API keys or credentials.
- Keep scraped data separate in `data/raw` and add sensitive sources to `.gitignore`.

License
- This repository uses the MIT License. See the LICENSE file for details.

Contact
- Open GitHub issues for bugs or feature requests.
- For model questions, open an issue with `model:` prefix and include sample inputs and expected outputs.

Images and badges
- Badges show main topics and link to related resources.
- Use the releases badge above to download packaged artifacts.

Second link reference for convenience:
https://github.com/JAMESWILLI2005/SpaceX/releases

Enjoy exploring model behavior and reuse economics with reproducible code, clear metrics, and modular pieces for production deployment.
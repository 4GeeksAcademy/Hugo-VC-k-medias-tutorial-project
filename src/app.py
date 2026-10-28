from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "train" / "housing.csv"
FEATURES = ["Latitude", "Longitude", "MedInc"]
N_CLUSTERS = 6


def main():
	housing = pd.read_csv(DATA_PATH)
	missing_columns = set(FEATURES) - set(housing.columns)
	if missing_columns:
		raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

	housing = housing[FEATURES].dropna().reset_index(drop=True)
	train_data, test_data = train_test_split(
		housing, test_size=0.2, random_state=42
	)

	kmeans = make_pipeline(
		StandardScaler(),
		KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10),
	)
	train_clusters = kmeans.fit_predict(train_data[FEATURES])
	test_clusters = kmeans.predict(test_data[FEATURES])

	classifier = make_pipeline(
		StandardScaler(),
		RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
	)
	classifier.fit(train_data[FEATURES], train_clusters)
	predicted_clusters = classifier.predict(test_data[FEATURES])

	train_data = train_data.assign(cluster=train_clusters)
	test_data = test_data.assign(cluster=test_clusters)
	test_data = test_data.assign(predicted_cluster=predicted_clusters)

	models_dir = ROOT_DIR / "models"
	output_dir = ROOT_DIR / "data" / "processed"
	models_dir.mkdir(parents=True, exist_ok=True)
	output_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(kmeans, models_dir / "kmeans.joblib")
	joblib.dump(classifier, models_dir / "cluster_classifier.joblib")
	train_data.to_csv(output_dir / "housing_train_clusters.csv", index=False)
	test_data.to_csv(output_dir / "housing_test_clusters.csv", index=False)

	scaler = kmeans.named_steps["standardscaler"]
	centers = scaler.inverse_transform(kmeans.named_steps["kmeans"].cluster_centers_)
	print("Centros de los clusters (Latitude, Longitude, MedInc):")
	print(pd.DataFrame(centers, columns=FEATURES).round(3).to_string(index=True))
	print("\nNúmero de viviendas por cluster en entrenamiento:")
	print(train_data["cluster"].value_counts().sort_index().to_string())
	print(f"\nExactitud del clasificador frente a K-Means: {accuracy_score(test_clusters, predicted_clusters):.3f}")
	print(
		classification_report(
			test_clusters,
			predicted_clusters,
			labels=range(N_CLUSTERS),
			zero_division=0,
		)
	)

	figure, axis = plt.subplots(figsize=(11, 7))
	axis.scatter(
		train_data["Longitude"],
		train_data["Latitude"],
		c=train_clusters,
		cmap="tab10",
		vmin=0,
		vmax=N_CLUSTERS - 1,
		s=9,
		alpha=0.55,
		label="Entrenamiento (K-Means)",
	)
	axis.scatter(
		test_data["Longitude"],
		test_data["Latitude"],
		c=predicted_clusters,
		cmap="tab10",
		vmin=0,
		vmax=N_CLUSTERS - 1,
		marker="x",
		s=14,
		alpha=0.65,
		label="Prueba (clasificador)",
	)
	axis.set(title="Agrupación de viviendas en California", xlabel="Longitud", ylabel="Latitud")
	axis.legend()
	figure.tight_layout()
	figure.savefig(output_dir / "housing_clusters.png", dpi=150)
	plt.close(figure)

	print(f"\nModelos guardados en: {models_dir}")
	print(f"Datos y gráfica guardados en: {output_dir}")


if __name__ == "__main__":
	main()

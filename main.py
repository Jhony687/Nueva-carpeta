from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import numpy as np
import pandas as pd

app = FastAPI()

class Pasajero(BaseModel):
    nombre: str
    telefono: str
    lat: float
    lng: float

@app.post("/agrupar")
async def agrupar_pasajeros(pasajeros: List[Pasajero]):
    df = pd.DataFrame([p.dict() for p in pasajeros])
    coords = df[['lat', 'lng']].apply(lambda row: (row['lat'], row['lng']), axis=1).tolist()

    def calcular_matriz_distancia_km(coords):
        size = len(coords)
        matriz = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                matriz[i][j] = great_circle(coords[i], coords[j]).kilometers
        return matriz

    distancias_km = calcular_matriz_distancia_km(coords)
    dbscan = DBSCAN(eps=4, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(distancias_km)

    df['grupo'] = labels

    grupos_resultado = []
    for grupo_id in sorted(df['grupo'].unique()):
        miembros = df[df['grupo'] == grupo_id]
        grupo_nombre = f"Grupo {grupo_id + 1}" if grupo_id != -1 else "Sin grupo"
        grupos_resultado.append({
            "grupo": grupo_nombre,
            "pasajeros": miembros[['nombre', 'telefono', 'lat', 'lng']].to_dict(orient='records')
        })

    return {"grupos": grupos_resultado}

from fastapi import FastAPI
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

    # Agrupamiento inicial con DBSCAN para encontrar grupos por cercanía (< 4km)
    dbscan = DBSCAN(eps=4, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(distancias_km)
    df['grupo_base'] = labels

    grupos_resultado = []
    grupo_contador = 1

    # Recorremos los grupos base para dividir en subgrupos de máximo 4
    for grupo_id in sorted(df['grupo_base'].unique()):
        miembros = df[df['grupo_base'] == grupo_id]
        if len(miembros) <= 4:
            grupos_resultado.append({
                "grupo": f"Grupo {grupo_contador}",
                "pasajeros": miembros[['nombre', 'telefono', 'lat', 'lng']].to_dict(orient='records')
            })
            grupo_contador += 1
        else:
            # Si hay más de 4, crear subgrupos de a máximo 4 personas
            subgrupos = [miembros[i:i+4] for i in range(0, len(miembros), 4)]
            for sub in subgrupos:
                grupos_resultado.append({
                    "grupo": f"Grupo {grupo_contador}",
                    "pasajeros": sub[['nombre', 'telefono', 'lat', 'lng']].to_dict(orient='records')
                })
                grupo_contador += 1

    return {"grupos": grupos_resultado}

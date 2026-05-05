import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ruta a tu archivo CSV local (ajusta si es necesario)
file_path = '/content/house_prices.csv' # Puedes subirlo a la sesión de Colab o especificar la ruta completa

try:
    df_house = pd.read_csv(file_path)
    print("Dataset cargado exitosamente.")
    print("Primeras 5 filas del dataset:")
    display(df_house.head())
    print("\nInformación general del dataset:")
    df_house.info()
except FileNotFoundError:
    print(f"Error: El archivo '{file_path}' no se encontró. Por favor, asegúrate de que el archivo esté subido o que la ruta sea correcta.")
except Exception as e:
    print(f"Ocurrió un error al cargar el archivo: {e}")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import joblib

def entrenar_modelos(df):

    # ======================
    # REGRESIÓN SIMPLE
    # ======================
    X_simple = df[['area']]
    y = df['price']

    model_simple = LinearRegression()
    model_simple.fit(X_simple, y)

    y_pred_simple = model_simple.predict(X_simple)

    print("=== REGRESIÓN SIMPLE ===")
    print("MSE:", mean_squared_error(y, y_pred_simple))
    print("R2:", r2_score(y, y_pred_simple))

    # 📊 Gráfica simple
    plt.scatter(df['area'], df['price'])
    plt.plot(df['area'], y_pred_simple, color='red')
    plt.title("Regresión Lineal Simple")
    plt.xlabel("Área")
    plt.ylabel("Precio")
    plt.show()


    # ======================
    # REGRESIÓN MÚLTIPLE
    # ======================
    X_multi = df[['area', 'rooms', 'age']]

    model_reg = LinearRegression()
    model_reg.fit(X_multi, y)

    y_pred_multi = model_reg.predict(X_multi)

    print("\n=== REGRESIÓN MÚLTIPLE ===")
    print("MSE:", mean_squared_error(y, y_pred_multi))
    print("R2:", r2_score(y, y_pred_multi))

    joblib.dump(model_reg, "modelo_regresion.pkl")


    # ======================
    # REGRESIÓN POLINÓMICA
    # ======================
    X_poly = df[['area']]

    poly = PolynomialFeatures(degree=2)
    X_poly_trans = poly.fit_transform(X_poly)

    model_poly = LinearRegression()
    model_poly.fit(X_poly_trans, y)

    y_pred_poly = model_poly.predict(X_poly_trans)

    print("\n=== REGRESIÓN POLINÓMICA ===")
    print("MSE:", mean_squared_error(y, y_pred_poly))
    print("R2:", r2_score(y, y_pred_poly))

    # 📊 Gráfica polinómica (curva)
    sorted_zip = sorted(zip(X_poly.values, y_pred_poly))
    X_sorted, y_poly_sorted = zip(*sorted_zip)

    plt.scatter(df['area'], df['price'])
    plt.plot(X_sorted, y_poly_sorted, color='green')
    plt.title("Regresión Polinómica")
    plt.xlabel("Área")
    plt.ylabel("Precio")
    plt.show()


    # ======================
    # CLASIFICACIÓN
    # ======================
    df['category'] = (df['price'] > df['price'].mean()).astype(int)

    X = df[['area', 'rooms', 'age']]
    y = df['category']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_clf = LogisticRegression()
    model_clf.fit(X_scaled, y)

    y_pred = model_clf.predict(X_scaled)

    print("\n=== CLASIFICACIÓN ===")
    print("Matriz de confusión:\n", confusion_matrix(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))

    joblib.dump(model_clf, "modelo_clasificacion.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
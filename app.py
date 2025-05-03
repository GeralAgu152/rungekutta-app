from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.pdfgen import canvas
import base64
import math as mt

app = Flask(__name__)

def runge_kutta4(f, y0, t0, tf, h):
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y

def f(t, y):
    return mt.exp(-t * y)

@app.route('/', methods=['GET', 'POST'])
def index():
    t = y = []
    img_data = ""
    if request.method == 'POST':
        y0 = float(request.form['y0'])
        t0 = float(request.form['t0'])
        tf = float(request.form['tf'])
        h = float(request.form['h'])

        if h <= 0 or tf <= t0:
            error_message = "Error: 'h' debe ser mayor que 0 y 'tf' debe ser mayor que 't0'."
            return render_template("index.html", error=error_message, t=[], y=[], img_data="")

        t, y = runge_kutta4(f, y0, t0, tf, h)

        # Gráfica
        fig, ax = plt.subplots()
        ax.plot(t, y, 'b')
        ax.set(xlabel='t', ylabel='y', title='Solución Runge-Kutta 4')
        ax.grid()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_data = base64.b64encode(img.read()).decode()
        plt.close()

    return render_template("index.html", t=t, y=y, img_data=img_data)

@app.route('/download', methods=['POST'])
def download():
    y0 = float(request.form['y0'])
    t0 = float(request.form['t0'])
    tf = float(request.form['tf'])
    h = float(request.form['h'])
    t, y = runge_kutta4(f, y0, t0, tf, h)

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 800, "Resultados del Método de Runge-Kutta de orden 4:")
    for i, (ti, yi) in enumerate(zip(t, y)):
        p.drawString(100, 780 - 15 * i, f"t = {ti:.2f}    y = {yi:.6f}")
        if 780 - 15 * i < 50:
            p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="resultados.pdf", mimetype='application/pdf')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)

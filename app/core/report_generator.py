import matplotlib.pyplot as plt
import os

def generate_bar_chart(metrics):
    strokes = [m["stroke"] for m in metrics]
    errors = [m["rmse"] for m in metrics]

    plt.figure()
    plt.bar(strokes, errors)
    plt.title("Stroke Error (RMSE)")
    plt.xlabel("Stroke")
    plt.ylabel("RMSE")
    plt.savefig("report.png")
    plt.close()

    return "report.png"
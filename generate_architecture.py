import matplotlib.pyplot as plt

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ======================
    # 🎨 Estilos (igual que el original)
    # ======================
    box_style_fe = dict(boxstyle='round,pad=0.5', facecolor='#e1f5fe', edgecolor='#0288d1', linewidth=2)
    box_style_be = dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    box_style_ai = dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#ef6c00', linewidth=2)
    arrow = dict(arrowstyle='->', color='#546e7a', linewidth=2)

    # ======================
    # 🏷️ Capas
    # ======================
    plt.text(0.5, 9.2, "CAPA DE PRESENTACIÓN (Frontend)", fontsize=14, fontweight='bold', color='#0288d1')
    plt.text(0.5, 6.2, "CAPA DE APLICACIÓN (Backend)", fontsize=14, fontweight='bold', color='#2e7d32')
    plt.text(0.5, 3.2, "CAPA DE INTELIGENCIA ARTIFICIAL", fontsize=14, fontweight='bold', color='#ef6c00')

    # ======================
    # 📐 GRID POSITIONS
    # ======================
    x_left, x_center, x_right = 3, 7, 11
    y_fe, y_be, y_ai = 8.3, 5.5, 2.0

    # ======================
    # 🧩 Componentes (SIN CAMBIOS)
    # ======================

    # Frontend
    ax.text(x_center, y_fe,
            "Interfaz de Usuario\n(HTML5 + CSS3 + Vanilla JS)\n[Fetch API / Progress Bar]",
            ha='center', va='center', bbox=box_style_fe)

    # Backend
    ax.text(x_center, y_be,
            "Servidor FastAPI (Uvicorn)\n[Manejo Asíncrono / Background Tasks]",
            ha='center', va='center', bbox=box_style_be)

    ax.text(x_left, y_be,
            "Almacenamiento Local\n(Videos / Gráficas PNG)",
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', edgecolor='#757575', linewidth=1))

    # IA
    ax.text(x_left, y_ai,
            "Visión por Computadora\n[OpenCV + Pandas]\n(Procesamiento de Frames)",
            ha='center', va='center', bbox=box_style_ai)

    ax.text(x_center, y_ai,
            "Inferencia YOLOv26\n[yolo26x.pt (Objetos)]\n[yolo26x-pose.pt (Pose)]",
            ha='center', va='center', bbox=box_style_ai)

    ax.text(x_right, y_ai,
            "Ollama LLM\n(llama3.2:3b)\n[Análisis & Resumen]",
            ha='center', va='center', bbox=box_style_ai)

    # ======================
    # 🔁 CONEXIONES (MEJORADAS)
    # ======================

    # 1. UI -> FastAPI (vertical)
    ax.annotate('', xy=(x_center, y_be + 0.6), xytext=(x_center, y_fe - 0.6), arrowprops=arrow)
    ax.text(7.7, 7.2, "1. POST /upload-video\n(Video Multipart)", fontsize=9)

    # 2. FastAPI -> Processing (L-shape)
    ax.annotate('', xy=(x_left, y_ai + 0.5), xytext=(x_center - 0.5, y_be - 0.5),
                arrowprops=dict(**arrow, connectionstyle="angle3,angleA=0,angleB=-90"))
    ax.text(2.8, 4.2, "2. Inicio Tarea\nBackground", fontsize=9)

    # 3. Processing -> YOLO (horizontal limpio)
    ax.annotate('', xy=(x_center - 0.7, y_ai), xytext=(x_left + 0.7, y_ai), arrowprops=arrow)
    ax.text(5, 2.3, "3. Detección", ha='center', fontsize=9)

    # 4. YOLO -> Processing (horizontal limpio)
    ax.annotate('', xy=(x_left + 0.7, y_ai - 0.3), xytext=(x_center - 0.7, y_ai - 0.3), arrowprops=arrow)
    ax.text(5, 1.4, "4. Keypoints &\nBoxes", ha='center', fontsize=9)

    # 5. Processing -> VLM (horizontal limpio)
    ax.annotate('', xy=(x_right - 0.7, y_ai), xytext=(x_center + 0.7, y_ai), arrowprops=arrow)
    ax.text(9, 2.2, "5. Key Frames\n(Base64) + Métricas", ha='center', fontsize=9)

    # 6. VLM -> Processing (L-shape limpio)
    ax.annotate('', xy=(x_center + 0.5, y_be - 0.3), xytext=(x_right, y_ai + 0.5),
                arrowprops=dict(**arrow, connectionstyle="angle3,angleA=180,angleB=90"))
    ax.text(8.2, 3.8, "6. Dictamen AI\nMarkdown", fontsize=9)

    # 7. Processing -> FastAPI (vertical limpio)
    ax.annotate('', xy=(x_center, y_be - 0.5), xytext=(x_center, y_ai + 0.5), arrowprops=arrow)
    ax.text(6.0, 3.8, "7. JSON Final\n(Task ID)", fontsize=9)

    # 8. FastAPI -> UI (vertical limpio)
    ax.annotate('', xy=(x_center + 0.5, y_fe - 0.5), xytext=(x_center + 0.5, y_be + 0.5), arrowprops=arrow)
    ax.text(7.7, 6.8, "8. Datos Análisis &\nVisualización", fontsize=9)

    # ======================
    # 🧠 Título
    # ======================
    plt.title("Arquitectura Actualizada: Tennis AI Analyzer Pro",
              fontsize=20, fontweight='bold', pad=30, color='#1a237e')

    plt.tight_layout()
    plt.savefig('arquitectura.png', dpi=300, bbox_inches='tight')
    print("✅ Arquitectura mejorada generada")


if __name__ == "__main__":
    create_architecture_diagram()
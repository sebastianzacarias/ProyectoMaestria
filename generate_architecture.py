import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Configuración de estilos
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='navy', linewidth=2)
    arrow_props = dict(arrowstyle='->', color='gray', linewidth=2, mutation_scale=20)
    
    # Capas (Layers)
    plt.text(0.5, 7.5, "Frontend (Web)", fontsize=14, fontweight='bold', color='navy')
    plt.text(0.5, 4.5, "Backend (FastAPI)", fontsize=14, fontweight='bold', color='darkgreen')
    plt.text(0.5, 1.5, "IA & Procesamiento", fontsize=14, fontweight='bold', color='darkred')

    # Componentes
    # 1. Frontend
    ax.text(6, 7, "Interfaz de Usuario\n(index.html / JavaScript)", ha='center', va='center', bbox=box_style)
    
    # 2. Backend
    ax.text(6, 4.5, "Servidor FastAPI\n(main.py)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e6f3ff', edgecolor='darkgreen', linewidth=2))
    
    # 3. Procesamiento
    ax.text(3, 2, "Video Processor\n(OpenCV & YOLO-Pose)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f0', edgecolor='darkred', linewidth=2))
    ax.text(6, 2, "YOLOv8\n(Detección de Objetos)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f0', edgecolor='darkred', linewidth=2))
    ax.text(9, 2, "Ollama VLM\n(qwen2-vl:8b)", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f0', edgecolor='darkred', linewidth=2))

    # Conexiones
    # De UI a FastAPI
    ax.annotate('', xy=(6, 5.2), xytext=(6, 6.2), arrowprops=arrow_props)
    ax.text(6.2, 5.7, "HTTP POST /upload-video", fontsize=9, color='gray')

    # De FastAPI a Video Processor
    ax.annotate('', xy=(3, 2.8), xytext=(5.2, 4.1), arrowprops=arrow_props)
    
    # De FastAPI a YOLO
    ax.annotate('', xy=(6, 2.8), xytext=(6, 3.8), arrowprops=arrow_props)

    # De FastAPI a Ollama
    ax.annotate('', xy=(9, 2.8), xytext=(6.8, 4.1), arrowprops=arrow_props)

    # De Procesadores a FastAPI (Respuestas)
    ax.text(3.5, 3.5, "Métricas Biomecánicas", rotation=30, fontsize=8, color='gray')
    ax.text(9.5, 3.5, "Resumen & Dictamen", rotation=-30, fontsize=8, color='gray')

    plt.title("Arquitectura del Sistema: Tennis AI Analyzer", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('arquitectura_solucion.png', dpi=300, bbox_inches='tight')
    print("Arquitectura generada exitosamente en arquitectura_solucion.png")

if __name__ == "__main__":
    create_architecture_diagram()

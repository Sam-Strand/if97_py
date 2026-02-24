import numpy as np
from PIL import Image
import io

# Ваш существующий код импорта и функций...

def create_rotating_gif(fig, output_filename='rotating_diagram.gif', duration=100, n_frames=72):
    """
    Создает вращающуюся GIF из 3D графика Plotly
    
    Parameters:
    fig: plotly figure
    output_filename: имя выходного файла
    duration: длительность каждого кадра в миллисекундах
    n_frames: количество кадров (по умолчанию 72 для плавного вращения)
    """
    frames = []
    
    # Создаем временную директорию для кадров, если её нет
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    print(f"Создание {n_frames} кадров...")
    
    for i in range(n_frames):
        # Вычисляем угол поворота
        angle = 360 * i / n_frames
        
        # Устанавливаем камеру для вращения вокруг вертикальной оси
        fig.update_layout(
            scene_camera=dict(
                eye=dict(
                    x=2.5 * np.sin(np.radians(angle)),
                    y=2.5 * np.cos(np.radians(angle)),
                    z=1.2
                ),
                center=dict(x=0, y=0, z=0)
            )
        )
        
        # Создаем изображение в памяти
        img_bytes = fig.to_image(format="png", width=1024, height=768, scale=1)
        
        # Конвертируем в PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        frames.append(img)
        
        if (i + 1) % 10 == 0:
            print(f"  Обработано {i + 1}/{n_frames} кадров")
    
    # Сохраняем как GIF
    print(f"Сохранение GIF в {output_filename}...")
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,  # 0 = бесконечный цикл
        optimize=False
    )
    
    # Очищаем временную директорию
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Готово! GIF сохранен как {output_filename}")
    return output_filename

# Ваш существующий код генерации данных...
print("=" * 60)
print("Создание 3D P-T-H диаграммы IAPWS IF97")
print("=" * 60)

num_points = 200  # Уменьшаем количество точек для более быстрой работы

print(f"\nГенерация {num_points:,} точек...")

from plot_3d import create_region_points, prepare_visualization_data, create_interactive_plot

# Создаем данные
T, P, H, dots = create_region_points(num_points)
print("\nПодготовка данных для визуализации...")
prepared_data = prepare_visualization_data(T, P, H)
# Создаем график (легкая операция с подготовленными данными)
print("\nСоздание интерактивной визуализации...")

print(f"\nРезультаты:")
print(f"  Всего точек: {len(T):,}")
print(f"  Температура: {T.min():.1f} - {T.max():.1f} K")
print(f"  Давление:    {P.min():.6f} - {P.max():.2f} MPa")
print(f"  Энтальпия:   {H.min():.1f} - {H.max():.1f} kJ/kg")

# Создаем график
print("\nСоздание интерактивной визуализации...")
fig = create_interactive_plot(T, P, H, dots, prepared_data)

# Создаем вращающуюся GIF
print("\nСоздание вращающейся GIF...")
gif_file = create_rotating_gif(
    fig, 
    output_filename='iapws_if97_rotating.gif',
    duration=150,  # 50 мс на кадр = 20 кадров в секунду
    n_frames=72   # 72 кадра для плавного вращения (5 секунд при 50 мс)
)

print(f"\nGIF создана: {gif_file}")

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Эта магия находит путь к текущему файлу, берет папку выше (корень) 
# и добавляет её в список мест, где Python ищет модули потому что иначе модули питона не работают
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from if97_py import bounds, steam, water, fluid, mix

from dataclasses import dataclass

@dataclass
class Dot:
    p: float
    t: float
    h: float
    text: str


def create_region_points(num_points: int = 200) -> tuple:
    """
    Создает точки для каждой области отдельно.
    Возвращает (T, P, H) для всех областей.
    """
    T_min, T_max = bounds.minT, bounds.maxT
    P_min, P_max = bounds.minP, bounds.maxP
    
    T_all, P_all, H_all = [], [], []
    
    def reg_4():
        # 1. Область 4 (пароводяная смесь) - по кривой насыщения
        print("Создание области 4 (пароводяная смесь)...")
        T_sat = np.linspace(T_min, bounds.t4Max, num_points)
        #T_sat = np.linspace(bounds.t4Max - 5, bounds.t4Max, 5)
        P_sat = bounds.saturationPressure_t(T_sat)
        #x_values = np.linspace(0, 1, 2)
        x_values = np.linspace(0, 1, 100)

        for t, p in zip(T_sat, P_sat):
            for x in x_values:
                h = mix.h.p_x(p, x)
                if h:
                    T_all.append(t)
                    P_all.append(p)
                    H_all.append(h)
    reg_4()

    def reg_1():
        print("Создание области 1 (вода)...")
        T_water = np.linspace(T_min, bounds.t3Min, num_points // 2)

        P_water = np.linspace(P_min, bounds.maxP, num_points // 2)
        
        T_water_grid, P_water_grid = np.meshgrid(T_water, P_water)
        T_water_flat = T_water_grid.flatten()
        P_water_flat = P_water_grid.flatten()
        
        H_water_flat = water.h.t_p(T_water_flat, P_water_flat)
        
        for t, p, h in zip(T_water_flat, P_water_flat, H_water_flat):
            if p >= bounds.minP and p <= bounds.maxP and t >= bounds.minT and t <= bounds.maxT:
                region = bounds.region_t_p(t, p)
                if region == 1:  # Вода
                    T_all.append(t)
                    P_all.append(p)
                    H_all.append(h)
                                 
    reg_1()

    def reg_2():
        print("Создание области 2 (перегретый пар)...")
        T_steam = np.linspace(bounds.minT, T_max, num_points // 2)
        
        P_steam_max = np.minimum(bounds.borderPressure_t(T_steam), P_max)
        P_steam_min = P_min + 1e-6
        
        P_steam_log = np.linspace(P_steam_min, P_steam_max.max(), num_points // 2)
        
        T_steam_grid, P_steam_log_grid = np.meshgrid(T_steam, P_steam_log)
        T_steam_flat = T_steam_grid.flatten()
        P_steam_flat = P_steam_log_grid.flatten()
        
        valid_mask = P_steam_flat < bounds.borderPressure_t(T_steam_flat)
        T_steam_valid = T_steam_flat[valid_mask]
        P_steam_valid = P_steam_flat[valid_mask]
        
        if len(T_steam_valid) > 0:
            H_steam_valid = steam.h.t_p(T_steam_valid, P_steam_valid)
            T_all.extend(T_steam_valid)
            P_all.extend(P_steam_valid)
            H_all.extend(H_steam_valid)
    reg_2()

    def reg_3():
        print("Создание области 3 (сверхкритическая жидкость)...")
        T_base = np.linspace(bounds.t3Min, T_max, num_points)
        P_base = np.linspace(bounds.p3Min, P_max, num_points)
        
        T_base_grid, P_base_grid = np.meshgrid(T_base, P_base)
        T_fluid = T_base_grid.flatten()
        P_fluid = P_base_grid.flatten()
        
        for t, p in zip(T_fluid, P_fluid):
            if p >= bounds.minP and p <= bounds.maxP and t >= bounds.minT and t <= bounds.maxT:
                region = bounds.region_t_p(t, p)
                if region == 3:  # Сверхкритическая жидкость
                    ρ = 1 / fluid.v.t_p(t, p)
                    h = fluid.h.t_ρ(t, ρ)
                    T_all.append(t)
                    P_all.append(p)
                    H_all.append(h)
    reg_3()

    dots = []
    # Критическая точка
    T_crit = bounds.t4Max
    P_crit = bounds.p4Max
    rho = fluid.v.t_p(T_crit, P_crit)
    H_crit = fluid.h.t_ρ(T_crit, 1 / rho)
    dots.append(Dot(
        p=P_crit,
        t=T_crit,
        h=H_crit,
        text=f"Критическая точка: T={T_crit}K"
    ))
    return np.array(T_all), np.array(P_all), np.array(H_all), dots


def create_interactive_plot(T, P, H, dots):
    """
    Создает интерактивную 3D визуализацию.
    """
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]],
        subplot_titles=['P-T-H Диаграмма (IAPWS IF97)']
    )
    def add_dot(p, t, h, text):
        fig.add_trace(go.Scatter3d(
            x=[t], y=[p], z=[h],
            mode='markers',
            marker=dict(size=8, color='black', symbol='diamond'),
            name=text
        ))
    
    colors = []
    regions = []
    for t, p in zip(T, P):
        region = bounds.region_t_p(t, p)
        if region == 1:    # вода
            colors.append('rgba(0, 0, 255, 0.7)') # синий
        elif region == 2:  # пар
            colors.append('rgba(255, 0, 0, 0.7)') # красный
        elif region == 3:  # сверхкритическая

            purple_palette = [
                'rgb(0, 0, 0)',
                'rgb(25, 25, 112)',
                'rgb(139, 0, 0)',
                'rgb(0, 100, 0)',
                'rgb(47, 79, 79)',
                'rgb(72, 61, 139)',
                'rgb(139, 69, 19)',
                'rgb(85, 107, 47)',
                'rgb(128, 0, 128)',
                'rgb(178, 34, 34)',
                'rgb(0, 0, 139)',
                'rgb(100, 0, 100)',
                'rgb(101, 67, 33)',
                'rgb(65, 105, 225)',
                'rgb(210, 105, 30)',
                'rgb(0, 128, 128)',
                'rgb(105, 105, 105)',
                'rgb(153, 50, 204)',
                'rgb(34, 139, 34)',
                'rgb(165, 42, 42)',
                'rgb(47, 50, 80)',
                'rgb(128, 70, 27)',
                'rgb(60, 20, 100)',
                'rgb(30, 60, 90)',
            ]

            subregion_color_map = {}
            region = fluid.v.get_sub_region(t, p)
            
            if region not in subregion_color_map:
                color_index = len(subregion_color_map) % len(purple_palette)
                subregion_color_map[region] = purple_palette[color_index]
            
            colors.append(subregion_color_map[region])
        elif region == 4:  # смесь
            colors.append('rgba(0, 128, 0, 0.7)') # зеленый
        else:
            colors.append('rgba(128, 128, 128, 0.7)') # иное (ошибки)
        regions.append(region)
        
    scatter = go.Scatter3d(
        text=regions,
        x=T, y=P, z=H,
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            opacity=1,
            symbol='circle'
        ),
        name='Термодинамические состояния',
        hovertemplate=(
            '<b>Температура</b>: %{x:.1f} K<br>' +
            '<b>Давление</b>: %{y:.4f} МПа<br>' +
            '<b>Энтальпия</b>: %{z:.1f} кДж/кг<br>' +
            '<b>Регион</b>: %{text}<br>' +
            '<extra></extra>'
        )
    )
    
    fig.add_trace(scatter)
    

    for dot in dots:
        add_dot(dot.p, dot.t, dot.h, dot.text)
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Температура, K',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            yaxis=dict(
                title='Давление, MPa',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            zaxis=dict(
                title='Энтальпия, kJ/kg',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        title=dict(
            text='3D P-T-H Диаграмма воды и водяного пара',
            x=0.5,
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1720,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Кнопки для видов
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(label="3D вид",
                         method="update",
                         args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2}}]),
                    dict(label="P-H проекция",
                         method="update",
                         args=[{"scene.camera.eye": {"x": 0, "y": -2.5, "z": 0}}]),
                    dict(label="T-H проекция",
                         method="update",
                         args=[{"scene.camera.eye": {"x": -2.5, "y": 0, "z": 0}}]),
                    dict(label="P-T проекция",
                         method="update",
                         args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}])
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )
    
    return fig


print("=" * 60)
print("Создание 3D P-T-H диаграммы IAPWS IF97")
print("=" * 60)

num_points = 500

print(f"\nГенерация {num_points:,} точек...")

# Создаем данные
T, P, H, dots = create_region_points(num_points)

print(f"\nРезультаты:")
print(f"  Всего точек: {len(T):,}")
print(f"  Температура: {T.min():.1f} - {T.max():.1f} K")
print(f"  Давление:    {P.min():.6f} - {P.max():.2f} MPa")
print(f"  Энтальпия:   {H.min():.1f} - {H.max():.1f} kJ/kg")


# Создаем график
print("\nСоздание интерактивной визуализации...")
fig = create_interactive_plot(T, P, H, dots)

print("Откройте файл в браузере для интерактивного просмотра")

# Показываем
fig.show()

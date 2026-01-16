import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Эта магия находит путь к текущему файлу, берет папку выше (корень) 
# и добавляет её в список мест, где Python ищет модули потому что иначе модули питона не работают
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from if97_py import bounds, steam, water, fluid

def create_region_points(num_points: int = 200) -> tuple:
    """
    Создает точки для каждой области отдельно.
    Возвращает (T, P, H) для всех областей.
    """
    T_min, T_max = bounds.minT, bounds.maxT
    P_min, P_max = bounds.minP, bounds.maxP
    
    T_all, P_all, H_all = [], [], []
    
    # 1. Область 4 (пароводяная смесь) - по кривой насыщения
    print("Создание области 4 (пароводяная смесь)...")
    T_sat_range = np.linspace(T_min, bounds.t4Max, num_points)
    P_sat = bounds.saturationPressure_t(T_sat_range)
    valid = ~np.isnan(P_sat)
    
    T_sat = T_sat_range[valid]
    P_sat = P_sat[valid]
    
    # Энтальпии границ
    H_water_sat = water.h.t_p(T_sat, P_sat)
    H_steam_sat = steam.h.t_p(T_sat, P_sat)
    
    # Добавляем точки смеси с разной степенью сухости
    x_values = np.linspace(0, 1, 100)  # 10 уровней сухости
    for t, p, h1, h2 in zip(T_sat, P_sat, H_water_sat, H_steam_sat):
        for x in x_values:
            h_mix = h1 + x * (h2 - h1)
            T_all.append(t)
            P_all.append(p)
            H_all.append(h_mix)
    
    # 2. Область 1 (вода) - равномерная сетка
    print("Создание области 1 (вода)...")
    T_water = np.linspace(T_min, bounds.t3Min, num_points // 2)
    # Для воды давление до p4Max
    P_water = np.logspace(np.log10(P_min), np.log10(bounds.p4Max), num_points // 2)
    
    T_water_grid, P_water_grid = np.meshgrid(T_water, P_water)
    T_water_flat = T_water_grid.flatten()
    P_water_flat = P_water_grid.flatten()
    
    # Вычисляем энтальпию для ВСЕХ точек
    H_water_flat = water.h.t_p(T_water_flat, P_water_flat)
    
    # Фильтруем только те, что действительно в области 1
    for t, p, h in zip(T_water_flat, P_water_flat, H_water_flat):
        if p >= bounds.minP and p <= bounds.maxP and t >= bounds.minT and t <= bounds.maxT:
            region = bounds.region_t_p(t, p)
            if region == 1:  # Вода
                T_all.append(t)
                P_all.append(p)
                H_all.append(h)
    
    # 3. Область 2 (перегретый пар) - равномерная сетка
    print("Создание области 2 (перегретый пар)...")
    T_steam = np.linspace(bounds.t3Min, T_max, num_points // 2)
    
    # Для каждого T вычисляем максимальное давление (граница с областью 1/3)
    P_steam_max = np.minimum(bounds.borderPressure_t(T_steam), P_max)
    P_steam_min = P_min + 1e-6
    
    # Создаем неравномерную сетку (больше точек при низких давлениях)
    P_steam_log = np.logspace(np.log10(P_steam_min), np.log10(P_steam_max.max()), num_points // 2)
    
    T_steam_grid, P_steam_log_grid = np.meshgrid(T_steam, P_steam_log)
    T_steam_flat = T_steam_grid.flatten()
    P_steam_flat = P_steam_log_grid.flatten()
    
    # Фильтруем: давление должно быть меньше граничного для данного T
    valid_mask = P_steam_flat < bounds.borderPressure_t(T_steam_flat)
    T_steam_valid = T_steam_flat[valid_mask]
    P_steam_valid = P_steam_flat[valid_mask]
    
    if len(T_steam_valid) > 0:
        H_steam_valid = steam.h.t_p(T_steam_valid, P_steam_valid)
        T_all.extend(T_steam_valid)
        P_all.extend(P_steam_valid)
        H_all.extend(H_steam_valid)
    
    # 4. Область 3 (сверхкритическая жидкость) - TODO: нужно добавить fluid модуль
    print("Создание области 3 (сверхкритическая жидкость)...")
    # Временное решение: используем steam.h.t_p, но это не совсем правильно
    T_fluid = np.linspace(bounds.t3Min, T_max, num_points // 3)
    P_fluid = np.logspace(np.log10(bounds.p3Min), np.log10(P_max), num_points // 3)
    
    T_fluid_grid, P_fluid_grid = np.meshgrid(T_fluid, P_fluid)
    T_fluid_flat = T_fluid_grid.flatten()
    P_fluid_flat = P_fluid_grid.flatten()
    
    # Проверяем, что точка в области 3
    for t, p in zip(T_fluid_flat, P_fluid_flat):
        if p >= bounds.minP and p <= bounds.maxP and t >= bounds.minT and t <= bounds.maxT:
            region = bounds.region_t_p(t, p)
            if region == 3:  # Сверхкритическая жидкость
                # Временное решение
                ρ = 1 / fluid.v.p_t(p, t)
                h = fluid.h.t_ρ(t, ρ)
                T_all.append(t)
                P_all.append(p)
                H_all.append(h)
    
    # 5. Граничные кривые
    print("Создание граничных кривых...")
    
    # Кривая насыщения (для визуализации)
    T_sat_dense = np.linspace(T_min, bounds.t4Max, 200)
    P_sat_dense = bounds.saturationPressure_t(T_sat_dense)
    valid_dense = ~np.isnan(P_sat_dense)
    
    if np.any(valid_dense):
        T_sat_curve = T_sat_dense[valid_dense]
        P_sat_curve = P_sat_dense[valid_dense]
        H_water_curve = water.h.t_p(T_sat_curve, P_sat_curve)
        H_steam_curve = steam.h.t_p(T_sat_curve, P_sat_curve)
        
        # Вода на кривой насыщения
        T_all.extend(T_sat_curve)
        P_all.extend(P_sat_curve)
        H_all.extend(H_water_curve)
        
        # Пар на кривой насыщения
        T_all.extend(T_sat_curve)
        P_all.extend(P_sat_curve)
        H_all.extend(H_steam_curve)
    
    # Граница 2-3
    T_border = np.linspace(bounds.t3Min, T_max, 100)
    P_border = bounds.borderPressure_t(T_border)
    valid_border = ~np.isnan(P_border)
    
    if np.any(valid_border):
        T_border_valid = T_border[valid_border]
        P_border_valid = P_border[valid_border]
        H_border = steam.h.t_p(T_border_valid, P_border_valid)
        
        T_all.extend(T_border_valid)
        P_all.extend(P_border_valid)
        H_all.extend(H_border)
    
    return np.array(T_all), np.array(P_all), np.array(H_all)


def create_interactive_plot(T, P, H):
    """
    Создает интерактивную 3D визуализацию.
    """
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]],
        subplot_titles=['P-T-H Диаграмма (IAPWS IF97)']
    )
    
    # Цвета по областям
    colors = []
    for t, p in zip(T, P):
        region = bounds.region_t_p(t, p)
        if region == 1:    # вода
            colors.append('rgba(0, 0, 255, 0.7)')
        elif region == 2:  # пар
            colors.append('rgba(255, 0, 0, 0.7)')
        elif region == 3:  # сверхкритическая
            colors.append('rgba(128, 0, 128, 0.7)')
        elif region == 4:  # смесь
            colors.append('rgba(0, 128, 0, 0.7)')
        else:
            colors.append('rgba(128, 128, 128, 0.7)')
    
    # Основные точки
    scatter = go.Scatter3d(
        x=T, y=P, z=H,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.6,
            symbol='circle'
        ),
        name='Термодинамические состояния',
        hovertemplate=(
            '<b>Температура</b>: %{x:.1f} K<br>' +
            '<b>Давление</b>: %{y:.4f} MPa<br>' +
            '<b>Энтальпия</b>: %{z:.1f} kJ/kg<br>' +
            '<extra></extra>'
        )
    )
    
    fig.add_trace(scatter)
    
    # Критическая точка
    T_crit = bounds.t4Max
    P_crit = bounds.p4Max
    H_crit = water.h.t_p(T_crit, P_crit)
    
    fig.add_trace(go.Scatter3d(
        x=[T_crit], y=[P_crit], z=[H_crit],
        mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='Критическая точка'
    ))
    
    # Настройки
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
                type='log',
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
        width=1400,
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

# Количество точек (можно увеличить до 10000+)
num_points = 500

print(f"\nГенерация {num_points:,} точек...")

# Создаем данные
T, P, H = create_region_points(num_points)

H = np.maximum(H, 0)

print(f"\nРезультаты:")
print(f"  Всего точек: {len(T):,}")
print(f"  Температура: {T.min():.1f} - {T.max():.1f} K")
print(f"  Давление:    {P.min():.6f} - {P.max():.2f} MPa")
print(f"  Энтальпия:   {H.min():.1f} - {H.max():.1f} kJ/kg")


# Создаем график
print("\nСоздание интерактивной визуализации...")
fig = create_interactive_plot(T, P, H)

# Сохраняем
output_file = "pt_h_diagram_3d.html"
fig.write_html(output_file)
print(f"\nДиаграмма сохранена: {output_file}")
print("Откройте файл в браузере для интерактивного просмотра")

# Показываем
fig.show()

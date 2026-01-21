import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Эта магия находит путь к текущему файлу, берет папку выше (корень) 
# и добавляет её в список мест, где Python ищет модули потому что иначе модули питона не работают
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from if97_py import bounds, steam, water, fluid, mix, consts

from dataclasses import dataclass

@dataclass
class Dot:
    p: float
    t: float
    h: float
    text: str

# Палитра для субрегионов
purple_palette = [
    'rgba(128, 0, 128, 0.8)',    # фиолетовый
    'rgba(186, 85, 211, 0.8)',   # средний фиолетовый
    'rgba(147, 112, 219, 0.8)',  # средне-фиолетовый
    'rgba(138, 43, 226, 0.8)',   # сине-фиолетовый
    'rgba(153, 50, 204, 0.8)',   # темный орхидей
    'rgba(148, 0, 211, 0.8)',    # темный фиолетовый
    'rgba(139, 0, 139, 0.8)',    # темный пурпурный
    'rgba(199, 21, 133, 0.8)',   # средний пурпурный
]

def add_boundary_lines(fig, num_points: int = 500, line_width: float = 4, opacity: float = 0.8):
    """
    Добавляет граничные линии на график.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Фигура для добавления линий
    num_points : int
        Количество точек для построения линий
    line_width : float
        Толщина линий
    opacity : float
        Прозрачность линий
    """
    
    def add_line(t_points, p_points, name, color, dash=None):
        """Добавляет одну линию на график"""
        # Рассчитываем энтальпию для каждой точки на линии
        h_points = []
        valid_t = []
        valid_p = []
        
        for t, p in zip(t_points, p_points):
            try:
                region = bounds.region_t_p(t, p)
                if region == 1:  # Вода
                    h = water.h.t_p(t, p)
                elif region == 2:  # Пар
                    h = steam.h.t_p(t, p)
                elif region == 3:  # Сверхкритическая
                    ρ = 1 / fluid.v.t_p(t, p)
                    h = fluid.h.t_ρ(t, ρ)
                elif region == 4:  # Смесь (двухфазная)
                    # Для линии насыщения используем насыщенный пар (x=1)
                    h = mix.h.p_x(p, 1)
                else:
                    continue
                    
                if h is not None:
                    h_points.append(h)
                    valid_t.append(t)
                    valid_p.append(p)
                    
            except Exception as e:
                continue
        
        if len(h_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=valid_t,
                y=valid_p,
                z=h_points,
                mode='lines',
                line=dict(
                    color=color,
                    width=line_width,
                    dash=dash
                ),
                name=name,
                hovertemplate=(
                    f'<b>{name}</b><br>' +
                    'Температура: %{x:.1f} K<br>' +
                    'Давление: %{y:.4f} МПа<br>' +
                    'Энтальпия: %{z:.1f} кДж/кг<br>' +
                    '<extra></extra>'
                ),
                opacity=opacity
            ))
    
    print("Добавление граничных линий...")
    
    # 1. Линия насыщения (reg_1_to_4 + reg_3_to_4 + reg_2_to_4)
    # Нижняя часть (вода -> насыщенный пар)
    p_values = np.linspace(consts.p3Min, consts.minP, num_points)
    t_sat = bounds.saturationTemp_p(p_values)
    add_line(t_sat, p_values, "Линия насыщения (вода-пар)", "black")
    
    # Верхняя часть (сверхкритическая -> насыщенный пар)
    p_values_upper = np.linspace(consts.p3Min, consts.p4Max, num_points)
    t_sat_upper = bounds.saturationTemp_p(p_values_upper)
    add_line(t_sat_upper, p_values_upper, "Линия насыщения (верхняя)", "black")
    
    # 2. Граница между областями 1 и 3 (reg_1_to_3)
    p_values_13 = np.linspace(consts.maxP, consts.p3Min, num_points)
    t_values_13 = np.full_like(p_values_13, consts.t3Min)
    add_line(t_values_13, p_values_13, "Граница 1-3", "black")
    
    # 3. Граница между областями 2 и 3 (reg_2_to_3)
    p_values_23 = np.linspace(consts.p3Min, consts.maxP, num_points)
    t_values_23 = bounds.borderTemp_p(p_values_23)
    add_line(t_values_23, p_values_23, "Граница 2-3", "black")
    
    # 4. Критическая точка
    t_crit = bounds.t4Max
    p_crit = bounds.p4Max
    rho_crit = 1 / fluid.v.t_p(t_crit, p_crit)
    h_crit = fluid.h.t_ρ(t_crit, rho_crit)
    
    fig.add_trace(go.Scatter3d(
        x=[t_crit],
        y=[p_crit],
        z=[h_crit],
        mode='markers+text',
        marker=dict(
            size=10,
            color='yellow',
            symbol='diamond',
            line=dict(width=2, color='black')
        ),
        text=["Критическая точка"],
        textposition="top center",
        name="Критическая точка",
        hovertemplate=(
            '<b>Критическая точка</b><br>' +
            f'Температура: {t_crit:.1f} K<br>' +
            f'Давление: {p_crit:.4f} МПа<br>' +
            f'Энтальпия: {h_crit:.1f} кДж/кг<br>' +
            '<extra></extra>'
        )
    ))

num_points = 500

# Граничные линии (старые списки - оставим для совместимости если нужно)
reg_1_to_3 = []
for p in np.linspace(consts.maxP, consts.p3Min, num_points):
    reg_1_to_3.append((consts.t3Min, p))

reg_1_to_4 = []
for p in np.linspace(consts.p3Min, consts.minP, num_points):
    t = bounds.saturationTemp_p(p)
    reg_1_to_4.append((t, p))

reg_2_to_3 = []
for p in np.linspace(consts.p3Min, consts.maxP, num_points):
    t = bounds.borderTemp_p(p)
    reg_2_to_3.append((t, p))

reg_2_to_4 = []
for p in np.linspace(consts.minP, consts.p3Min, num_points):
    t = bounds.saturationTemp_p(p)
    reg_2_to_4.append((t, p))

reg_3_to_4 = []
for p in np.linspace(consts.p3Min, consts.p4Max, num_points):
    t = bounds.saturationTemp_p(p)
    reg_3_to_4.append((t, p))


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
        P_sat = bounds.saturationPressure_t(T_sat)
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
        # 2. Область 1 (вода) - равномерная сетка
        print("Создание области 1 (вода)...")
        T_water = np.linspace(T_min, bounds.t3Min, num_points // 2)
        # Для воды давление до p4Max
        P_water = np.linspace(P_min, bounds.maxP, num_points // 2)
        
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
                                 
    reg_1()

    def reg_2():
        # 3. Область 2 (перегретый пар) - равномерная сетка
        print("Создание области 2 (перегретый пар)...")
        T_steam = np.linspace(bounds.minT, T_max, num_points // 2)
        
        # Для каждого T вычисляем максимальное давление (граница с областью 1/3)
        P_steam_max = np.minimum(bounds.borderPressure_t(T_steam), P_max)
        P_steam_min = P_min + 1e-6
        
        # Создаем неравномерную сетку (больше точек при низких давлениях)
        P_steam_log = np.linspace(P_steam_min, P_steam_max.max(), num_points // 2)
        
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
    reg_2()

    def reg_3():
        print("Создание области 3 (сверхкритическая жидкость)...")
        T_fluid = np.linspace(bounds.t3Min, T_max, num_points // 1)
        P_fluid = np.linspace(bounds.p3Min, P_max, num_points // 1)
        
        T_fluid_grid, P_fluid_grid = np.meshgrid(T_fluid, P_fluid)
        T_fluid_flat = T_fluid_grid.flatten()
        P_fluid_flat = P_fluid_grid.flatten()
        
        # Проверяем, что точка в области 3
        for t, p in zip(T_fluid_flat, P_fluid_flat):
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
    print('rho', rho)
    H_crit = fluid.h.t_ρ(T_crit, 1 / rho)
    dots.append(Dot(
        p=P_crit,
        t=T_crit,
        h=H_crit,
        text=f"Критическая точка: T={T_crit}K"
    ))
    return np.array(T_all), np.array(P_all), np.array(H_all), dots


# Маппинг субрегионов в цвета
subregion_color_map = {}

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
    
    # Цвета по областям
    colors = []
    regions = []
    for t, p in zip(T, P):
        region = bounds.region_t_p(t, p)
        if region == 1:    # вода
            colors.append('rgba(0, 0, 255, 0.7)') # синий
        elif region == 2:  # пар
            colors.append('rgba(255, 0, 0, 0.7)') # красный
        elif region == 3:  # сверхкритическая
            sub_region = fluid.v.get_sub_region(t, p)
            
            # Создаем маппинг при первом появлении субрегиона
            if sub_region not in subregion_color_map:
                # Берем следующий доступный цвет из палитры
                color_index = len(subregion_color_map) % len(purple_palette)
                subregion_color_map[sub_region] = purple_palette[color_index]
            
            colors.append(subregion_color_map[sub_region])
        elif region == 4:  # смесь
            colors.append('rgba(0, 128, 0, 0.7)') # зеленый
        else:
            colors.append('rgba(128, 128, 128, 0.7)') # иное (ошибки)
        regions.append(region)
    
    # Основные точки
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
    
    # Добавляем граничные линии
    add_boundary_lines(fig, num_points=200, line_width=3, opacity=0.9)

    for dot in dots:
        add_dot(dot.p, dot.t, dot.h, dot.text)
    
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
    

    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='right',
                x=0.0,
                y=1.15,
                buttons=[
                    dict(
                        label='Вид сверху',
                        method='relayout',
                        args=[{
                            'scene.camera': {
                                'projection':dict(type="orthographic"),
                                #'up':dict(x=0, y=1, z=0),
                                'eye': {'x': 0, 'y': 0, 'z': 2}
                            }
                        }]
                    ),
                    dict(
                        label='Изометрия',
                        method='relayout',
                        args=[{
                            'scene.camera': {
                                'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25}
                            }
                        }]
                    )
                ]
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
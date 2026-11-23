'''
Проверка реализации по контрольным значениям IAPWS-IF97.

Запуск:
    python tests/test_if97.py        # без зависимостей, печатает отчёт
    pytest tests/test_if97.py        # если нужен pytest

Источники контрольных значений (IF97-Rev):
    Таблица  5  — область 1, v/h/u/s/cp/w
    Таблица 15  — область 2, то же
    Таблица 33  — область 3, p/h/u/s/cp/w по T и rho
    Таблица 42  — область 5, то же что 5 и 15
    Таблица 35  — давление насыщения
    Таблица 36  — температура насыщения
    Таблица  7  — область 1, обратное T(p,h)
    Таблица  9  — область 1, обратное T(p,s)
    Таблица 24  — область 2, обратное T(p,h), подобласти 2a/2b/2c
    Таблица 29  — область 2, обратное T(p,s), подобласти 2a/2b/2c
    стр. 6      — контрольная точка уравнения B23
    стр. 22     — контрольная точка уравнения B2bc

Помимо таблиц проверяются внутренние согласованности, которых в стандарте нет,
но которые ловят ошибки в производных и в выборе подобласти:
    rho * v == 1 во всех областях;
    p, восстановленное из уравнения Гельмгольца по v(p,T) области 3;
    критерий Максвелла g' == g'' на линии насыщения;
    непрерывность x(p,h) и x(p,s) относительно h(p,x) и s(p,x);
    совпадение скалярного и массивного путей;
    круговой обход T(p, h(p,T)) и T(p, s(p,T)) против допуска 25 мК,
        который стандарт сам отводит обратным уравнениям;
    выбор подобласти 2a/2b/2c против прямого вызова её уравнения.
'''

import numpy as np

from if97 import bounds, if97

# Контрольные значения даны в стандарте девятью значащими цифрами
TOL = 2e-8


def _rel(got, ref):
    return float(np.abs(np.asarray(got, float) / np.asarray(ref, float) - 1).max())


def _check(label, got, ref, tol=TOL):
    r = _rel(got, ref)
    assert r <= tol, f'{label}: отн. расхождение {r:.3e} > {tol:.0e}'
    return r


# --------------------------------------------------------------------------
# Контрольные значения стандарта
# --------------------------------------------------------------------------

def test_region1_table5():
    t = np.array([300., 300., 500.])
    p = np.array([3., 80., 3.])
    s = if97.water.state.t_p(t, p)
    _check('r1 v', s.v, (0.100215168e-2, 0.971180894e-3, 0.120241800e-2))
    _check('r1 h', s.h, (0.115331273e3, 0.184142828e3, 0.975542239e3))
    _check('r1 u', s.u, (0.112324818e3, 0.106448356e3, 0.971934985e3))
    _check('r1 s', s.s, (0.392294792, 0.368563852, 0.258041912e1))
    _check('r1 cp', s.cp, (0.417301218e1, 0.401008987e1, 0.465580682e1))
    _check('r1 w', s.w, (0.150773921e4, 0.163469054e4, 0.124071337e4))


def test_region2_table15():
    t = np.array([300., 700., 700.])
    p = np.array([0.0035, 0.0035, 30.])
    s = if97.steam.state.t_p(t, p)
    _check('r2 v', s.v, (0.394913866e2, 0.923015898e2, 0.542946619e-2))
    _check('r2 h', s.h, (0.254991145e4, 0.333568375e4, 0.263149474e4))
    _check('r2 u', s.u, (0.241169160e4, 0.301262819e4, 0.246861076e4))
    _check('r2 s', s.s, (0.852238967e1, 0.101749996e2, 0.517540298e1))
    _check('r2 cp', s.cp, (0.191300162e1, 0.208141274e1, 0.103505092e2))
    _check('r2 w', s.w, (0.427920172e3, 0.644289068e3, 0.480386523e3))


def test_region3_table33():
    t = np.array([650., 650., 750.])
    rho = np.array([500., 200., 500.])
    s = if97.fluid.state.t_rho(t, rho)
    _check('r3 p', if97.fluid.p.t_rho(t, rho),
           (0.255837018e2, 0.222930643e2, 0.783095639e2))
    _check('r3 h', s.h, (0.186343019e4, 0.237512401e4, 0.225868845e4))
    _check('r3 u', s.u, (0.181226279e4, 0.226365868e4, 0.210206932e4))
    _check('r3 s', s.s, (0.405427273e1, 0.485438792e1, 0.446971906e1))
    _check('r3 cp', s.cp, (0.138935717e2, 0.446579342e2, 0.634165359e1))
    _check('r3 w', s.w, (0.502005554e3, 0.383444594e3, 0.760696041e3))


def test_region5_table42():
    t = np.array([1500., 1500., 2000.])
    p = np.array([0.5, 30., 30.])
    s = if97.gas.state.t_p(t, p)
    _check('r5 v', s.v, (0.138455090e1, 0.230761299e-1, 0.311385219e-1))
    _check('r5 h', s.h, (0.521976855e4, 0.516723514e4, 0.657122604e4))
    _check('r5 u', s.u, (0.452749310e4, 0.447495124e4, 0.563707038e4))
    _check('r5 s', s.s, (0.965408875e1, 0.772970133e1, 0.853640523e1))
    _check('r5 cp', s.cp, (0.261609445e1, 0.272724317e1, 0.288569882e1))
    _check('r5 w', s.w, (0.917068690e3, 0.928548002e3, 0.106736948e4))


def test_saturation_tables35_36():
    _check('p_s(T)', bounds.saturationPressure_t(np.array([300., 500., 600.])),
           (0.353658941e-2, 0.263889776e1, 0.123443146e2))
    _check('T_s(p)', bounds.saturationTemp_p(np.array([0.1, 1., 10.])),
           (0.372755919e3, 0.453035632e3, 0.584149488e3))


def test_b23_checkpoint():
    _check('B23 p(T)', bounds.b23Pressure_t(0.623150000e3), 0.165291643e2)
    _check('B23 T(p)', bounds.b23Temp_p(0.165291643e2), 0.623150000e3)


# --------------------------------------------------------------------------
# Диапазоны областей
# --------------------------------------------------------------------------

def test_region_detection():
    cases = (
        (300., 3., 1.), (300., 80., 1.), (300., 100., 1.),   # до 100 МПа
        (700., 0.0035, 2.), (1000., 10., 2.),
        (650., 30., 3.), (630., 80., 3.),
        (1500., 30., 5.), (1200., 0.1, 5.),
    )
    for t, p, want in cases:
        got = bounds.region_t_p(t, p)
        assert got == want, f'region_t_p({t}, {p}) = {got}, ожидалось {want}'
    for t, p in ((300., 100.001), (2300., 1.), (1500., 60.), (250., 1.)):
        assert np.isnan(bounds.region_t_p(t, p)), f'ожидался NaN при T={t}, p={p}'


def test_region5_pressure_ceiling():
    # область 5 ограничена 50 МПа, области 1-3 идут до 100
    assert bounds.region_t_p(1500., 50.) == 5.
    assert np.isnan(bounds.region_t_p(1500., 50.001))
    assert bounds.region_t_p(300., 100.) == 1.


# --------------------------------------------------------------------------
# Внутренние согласованности
# --------------------------------------------------------------------------

def _grid(region, t_lo, t_hi, p_lo, p_hi, n=60):
    t, p = np.meshgrid(np.linspace(t_lo, t_hi, n), np.linspace(p_lo, p_hi, n))
    t, p = t.ravel(), p.ravel()
    m = bounds.region_t_p(t, p) == region
    return t[m], p[m]


def test_rho_v_reciprocal():
    for mod, reg, box in ((if97.water, 1, (300., 620., 20., 50.)),
                          (if97.steam, 2, (700., 1000., 0.5, 10.)),
                          (if97.fluid, 3, (630., 700., 25., 45.)),
                          (if97.gas, 5, (1100., 2200., 1., 50.))):
        t, p = _grid(reg, *box)
        s = mod.state.t_p(t, p)
        err = float(np.abs(s.rho * s.v - 1.0).max())
        assert err < 1e-12, f'область {reg}: rho*v - 1 = {err:.2e}'


def test_region3_backward_roundtrip():
    '''v(p,T) обратного уравнения, пропущенный через уравнение Гельмгольца.'''
    t, p = _grid(3, 623.2, 860., 17., 100., n=120)
    v = if97.fluid.v.t_p(t, p)
    pb = if97.fluid.p.t_rho(t, 1.0 / v)
    rel = np.abs(pb / p - 1.0)
    # вблизи критической точки dp/dv -> 0 и допуск шире (SR5-05, §5)
    near = (np.abs(p - 22.064) < 1.0) & (np.abs(t - 647.096) < 5.0)
    assert rel[~near].max() < 1e-4, f'вдали от крит. точки: {rel[~near].max():.2e}'
    assert rel.max() < 1e-3, f'включая крит. точку: {rel.max():.2e}'


def test_region3_all_subregions_reached():
    t, p = _grid(3, 623.2, 860., 17., 100., n=200)
    t2, p2 = _grid(3, 645., 650.5, 21.9, 22.3, n=200)
    seen = set(np.concatenate([if97.fluid.get_sub_region(t, p),
                               if97.fluid.get_sub_region(t2, p2)]).tolist())
    assert len(seen) >= 24, f'задействовано подобластей: {len(seen)} из 26'


def test_saturation_maxwell():
    '''
    Условие фазового равновесия g' == g'' (Таблица 31 стандарта).

    Ловит ошибку, из-за которой энтропия смеси выше 16.53 МПа считалась по
    уравнениям областей 1 и 2, тогда как обе фазы там лежат в области 3.
    '''
    for p in (0.01, 0.1, 1., 5., 10., 15., 17., 20., 22.):
        t = bounds.saturationTemp_p(p)
        liq = if97.mix.state.p_x(p, 0.0)
        vap = if97.mix.state.p_x(p, 1.0)
        g1 = liq.h - t * liq.s
        g2 = vap.h - t * vap.s
        # допустимая невязка по g на линии насыщения — 0.2 кДж/кг (Таблица 43)
        assert abs(g1 - g2) < 0.2, f'p={p}: g\'-g\'\' = {g1 - g2:.4f} кДж/кг'


def test_saturation_phases_match_single_phase_regions():
    '''Ниже p3Min насыщенные фазы должны совпадать с областями 1 и 2.'''
    for p in (0.01, 0.1, 1., 5., 10., 15., 16.5):
        t = bounds.saturationTemp_p(p)
        liq = if97.mix.state.p_x(p, 0.0)
        vap = if97.mix.state.p_x(p, 1.0)
        for f in ('v', 'h', 's', 'u'):
            _check(f"{f}' при p={p}", getattr(liq, f),
                   getattr(if97.water.state.t_p(t, p), f), 1e-10)
            _check(f'{f}" при p={p}', getattr(vap, f),
                   getattr(if97.steam.state.t_p(t, p), f), 1e-10)


def test_mixture_endpoints_and_roundtrip():
    for p in (0.01, 0.1, 1., 5., 10., 15., 16., 18., 20., 22.):
        t = bounds.saturationTemp_p(p)
        if p <= 16.5:
            _check(f'x=0 при p={p}', if97.mix.v.p_x(p, 0.0),
                   if97.water.v.t_p(t, p), 1e-12)
            _check(f'x=1 при p={p}', if97.mix.v.p_x(p, 1.0),
                   if97.steam.v.t_p(t, p), 1e-12)
        for x in (0.0, 0.25, 0.5, 0.9, 1.0):
            m = if97.mix.state.p_x(p, x)
            assert abs(if97.mix.x.p_h(p, m.h) - x) < 1e-12
            assert abs(if97.mix.x.p_s(p, m.s) - x) < 1e-12


def test_mixture_outside_two_phase_is_nan():
    assert np.isnan(if97.mix.x.t_h(700., 2000.))   # выше критической температуры
    assert np.isnan(if97.mix.x.p_h(30., 2000.))    # выше критического давления
    assert np.isnan(if97.mix.x.p_s(30., 5.))


def test_out_of_domain_returns_nan_everywhere():
    """
    Вне области определения возвращается NaN, а не исключение и не мусор.

    Проверяются и скалярный путь, и пакетный: исключение из середины цикла
    по массиву — худшее, что может случиться с расчётом свойств.

    Тест стережёт fastmath. Флаг nnan сообщает LLVM, что NaN не бывает, и под
    ним проверки вида `x != x` сворачиваются в False, а арифметика с NaN
    становится неопределённой. Функции, которые NaN производят или проверяют,
    обязаны компилироваться без него (профиль JIT_NAN).
    """
    cases = (
        ('p_s выше Tc', if97.saturationPressure_t, (700.,)),
        ('p_s ниже Tmin', if97.saturationPressure_t, (200.,)),
        ('T_s выше pc', if97.saturationTemp_p, (30.,)),
        ('T_s ниже pmin', if97.saturationTemp_p, (1e-9,)),
        ('x(T,h) выше Tc', if97.mix.x.t_h, (700., 2000.)),
        ('x(T,s) выше Tc', if97.mix.x.t_s, (700., 5.)),
        ('x(p,h) выше pc', if97.mix.x.p_h, (30., 2000.)),
        ('x(p,s) выше pc', if97.mix.x.p_s, (30., 5.)),
        ('h смеси выше pc', if97.mix.h.p_x, (30., 0.5)),
        ('v смеси выше Tc', if97.mix.v.t_x, (700., 0.5)),
        ('состояние смеси выше pc', lambda a, b: if97.mix.state.p_x(a, b).h, (30., 0.5)),
        ('область вне диапазона', if97.region_t_p, (300., 200.)),
        ('авто вне диапазона', if97.h_t_p, (2300., 1.)),
        ('авто выше 50 МПа в обл. 5', if97.h_t_p, (1500., 60.)),
    )
    for label, fn, args in cases:
        got = fn(*args)
        assert np.isnan(got), f'{label}: скаляр вернул {got!r}, ожидался NaN'
        arr = fn(*(np.full(4, a) for a in args))
        assert np.all(np.isnan(arr)), f'{label}: массив вернул {arr!r}'


def test_nan_mixed_with_valid_in_batch():
    """В массиве NaN не должен портить соседние точки."""
    p = np.array([1.0, 30.0, 5.0, 100.0, 0.1])      # 30 и 100 выше критического
    x = np.full(p.size, 0.4)
    h = if97.mix.h.p_x(p, x)
    assert np.isnan(h[[1, 3]]).all(), h
    assert np.isfinite(h[[0, 2, 4]]).all(), h
    # те же точки по одной дают то же самое
    for i, pi in enumerate(p):
        one = if97.mix.h.p_x(float(pi), 0.4)
        assert (np.isnan(one) and np.isnan(h[i])) or one == h[i], (i, one, h[i])


def test_nan_is_a_gap_not_an_error():
    '''
    `nan` во входе — пропуск в ряду, а не ошибка расчёта.

    Временные ряды приходят с пропусками, и раньше любая `nan` роняла расчёт
    (в том числе `ZeroDivisionError` из ядра пара). Теперь точки с пропуском
    в ядро не идут: у них в выходе `nan`, остальные считаются как обычно.
    '''
    nan = float('nan')

    # Точечные вызовы: nan на входе — nan на выходе, без исключений
    for label, value in (('h пара', if97.steam.h.t_p(nan, 13.0)),
                         ('h пара по nan-давлению', if97.steam.h.t_p(821.0, nan)),
                         ('температура насыщения', if97.saturationTemp_p(nan))):
        assert value != value, f'{label}: ожидался nan, получено {value!r}'

    # Пакетный вызов: пропуски остаются на своих местах, остальное считается
    t = np.array([nan, nan, 821.0, 700.0])
    p = np.array([13.0, 13.0, 13.0, nan])
    h = if97.steam.h.t_p(t, p)
    assert h[0] != h[0] and h[1] != h[1] and h[3] != h[3], f'пропуски потеряны: {h}'
    _check('h пара по конечной точке', h[2], if97.steam.h.t_p(821.0, 13.0), 1e-13)

    # Все точки — пропуски: пустой результат, не исключение
    empty = if97.steam.h.t_p(np.full(3, nan), 13.0)
    assert np.isnan(empty).all(), f'ожидался ряд из nan, получено {empty}'


def test_scalar_matches_array():
    """
    Точечный расчёт совпадает с пакетным.

    Побитово — не обязательно: скалярный вход и цикл это две отдельные
    компиляции одного ядра, и под fastmath LLVM в цикле переассоциирует
    сумму иначе, чем в одиночном вызове. Расхождение в последнем бите
    нормально; стандарт даёт контрольные значения девятью цифрами.
    """
    t = np.array([300., 700., 650., 1500.])
    p = np.array([3., 1., 30., 30.])
    arr = if97.state_t_p(t, p)
    for i, (ti, pi) in enumerate(zip(t, p)):
        sc = if97.state_t_p(float(ti), float(pi))
        for f in sc._fields:
            _check(f'{f} при T={ti}, p={pi}', getattr(sc, f), getattr(arr, f)[i], 1e-13)


def test_lazy_imports():
    '''
    Обращение к области поднимает только её.

    Раскладка по файлам существует ради этого: тронул water — компилируется
    область 1, а не дерево из 26 подобластей области 3 и область 5 заодно.
    '''
    import subprocess
    import sys

    probe = (
        'import sys\n'
        'AREAS = ("if97.water", "if97.steam", "if97.fluid", "if97.gas",\n'
        '         "if97.mix", "if97.auto", "if97.bounds")\n'
        'from if97 import if97\n'
        'early = [m for m in sys.modules if m in AREAS]\n'
        'assert not early, early          # import if97 ничего не компилирует\n'
        'if97.water.h.t_p(300., 3.)\n'
        'loaded = {m for m in sys.modules if m.startswith("if97.")}\n'
        'for heavy in ("if97.fluid", "if97.gas", "if97.mix", "if97.steam"):\n'
        '    assert heavy not in loaded, (heavy, sorted(loaded))\n'
        'print("ok")\n'
    )
    r = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr.strip()


def test_broadcasting():
    h = if97.h_t_p(np.array([300., 400., 500.]), 3.0)
    assert h.shape == (3,)
    assert abs(h[0] / 0.115331273e3 - 1) < TOL


def test_auto_dispatch_matches_direct():
    """
    Автовыбор области даёт то же, что прямое обращение к области.

    Побитового совпадения здесь ждать нельзя: auto.py компилируется с
    выключенным fastmath (он сравнивает код области, а тот бывает NaN),
    а ядра областей — с включённым. При инлайне LLVM по-разному решает
    про свёртку в FMA, и последний бит расходится. Допуск на одиннадцати
    знаках ловит любую содержательную ошибку и игнорирует это.
    """
    for reg, mod, box in ((1, if97.water, (300., 620., 20., 50.)),
                          (2, if97.steam, (700., 1000., 0.5, 10.)),
                          (3, if97.fluid, (630., 670., 30., 45.)),
                          (5, if97.gas, (1100., 2200., 1., 50.))):
        t, p = _grid(reg, *box)
        a = if97.state_t_p(t, p)
        b = mod.state.t_p(t, p)
        for f in a._fields:
            _check(f'область {reg}, {f}', getattr(a, f), getattr(b, f), 1e-11)


# --------------------------------------------------------------------------
# Обратные уравнения T(p,h) и T(p,s), области 1 и 2
# --------------------------------------------------------------------------

#: Допуск самого стандарта на обратные уравнения, [K]: уравнения (12), (14),
#: (28) и (30). Это подгонки, а не обращения, и точнее они быть не обязаны.
DT_TOL = 0.025


def test_backward_region1_tables7_9():
    p = np.array([3., 80., 80.])
    _check('r1 T(p,h)', if97.water.t.p_h(p, np.array([500., 500., 1500.])),
           (0.391798509e3, 0.378108626e3, 0.611041229e3))
    _check('r1 T(p,s)', if97.water.t.p_s(p, np.array([0.5, 0.5, 3.])),
           (0.307842258e3, 0.309979785e3, 0.565899909e3))


def test_backward_region2_tables24_29():
    t = if97.steam.t
    _check('r2a T(p,h)', t.p_h(np.array([0.001, 3., 3.]), np.array([3000., 3000., 4000.])),
           (0.534433241e3, 0.575373370e3, 0.101077577e4))
    _check('r2b T(p,h)', t.p_h(np.array([5., 5., 25.]), np.array([3500., 4000., 3500.])),
           (0.801299102e3, 0.101531583e4, 0.875279054e3))
    _check('r2c T(p,h)', t.p_h(np.array([40., 60., 60.]), np.array([2700., 2700., 3200.])),
           (0.743056411e3, 0.791137067e3, 0.882756860e3))
    _check('r2a T(p,s)', t.p_s(np.array([0.1, 0.1, 2.5]), np.array([7.5, 8., 8.])),
           (0.399517097e3, 0.514127081e3, 0.103984917e4))
    _check('r2b T(p,s)', t.p_s(np.array([8., 8., 90.]), np.array([6., 7.5, 6.])),
           (0.600484040e3, 0.106495556e4, 0.103801126e4))
    _check('r2c T(p,s)', t.p_s(np.array([20., 80., 80.]), np.array([5.75, 5.25, 5.75])),
           (0.697992849e3, 0.854011484e3, 0.949017998e3))


def test_b2bc_checkpoint():
    # Контрольная точка со стр. 22 плюс согласие уравнений (20) и (21).
    h_p, p_h = bounds.b2bcEnthalpy_p, bounds.b2bcPressure_h
    _check('B2bc h(p)', h_p(0.100000000e3), 0.3516004323e4)
    _check('B2bc p(h)', p_h(0.3516004323e4), 0.100000000e3)
    p = np.linspace(bounds.p2bcMin, 100., 50)
    _check('B2bc p(h(p))', p_h(h_p(p)), p, 1e-9)
    # ниже PMIN линии нет: уравнение (21) должно отдать NaN, а не мусор
    assert np.isnan(h_p(4.1)), 'B2bc ниже PMIN: ожидался NaN для скаляра'
    assert np.all(np.isnan(h_p(np.array([1., 4.1, 4.5])))), \
        'B2bc ниже PMIN: ожидался NaN для массива'


def test_backward_roundtrip_within_tolerance():
    # |T(p, h(p,T)) - T| и то же по энтропии против допуска стандарта.
    for reg, mod, box in ((1, if97.water, (275., 620., 1., 100.)),
                          (2, if97.steam, (275., 1070., 1e-3, 100.))):
        t, p = _grid(reg, *box, n=120)
        for name, prop in (('h', mod.h), ('s', mod.s)):
            back = getattr(mod.t, f'p_{name}')
            d = float(np.abs(back(p, prop.t_p(t, p)) - t).max())
            assert d <= DT_TOL, (f'область {reg}, T(p,{name}): отклонение '
                                 f'{d * 1e3:.1f} мК > {DT_TOL * 1e3:.0f} мК')


def test_backward_region2_subregions_dispatch():
    # Все три подобласти задеты, и выбор ветви совпадает с прямым вызовом.
    t = if97.steam.t
    tb = if97.steam.tables_backward
    pw_x, pw_y = t.scratch()
    tg, pg = _grid(2, 275., 1070., 1e-3, 100., n=120)
    h = if97.steam.h.t_p(tg, pg)
    s = if97.steam.s.t_p(tg, pg)

    above = pg >= t.PMIN
    a = pg <= t.P_2AB
    # h_p спрашивается только там, где линия B2bc существует
    c_h = ~a & above & (h < bounds.b2bcEnthalpy_p(np.where(above, pg, t.PMIN)))
    b_h = ~a & ~c_h
    c_s = ~a & (s < t.S_2BC)
    b_s = ~a & ~c_s

    for nm, mask, core, tab, y in (
            ('2a T(p,h)', a, t.ah_core, (tb.AH_I, tb.AH_J, tb.AH_N), h),
            ('2b T(p,h)', b_h, t.bh_core, (tb.BH_I, tb.BH_J, tb.BH_N), h),
            ('2c T(p,h)', c_h, t.ch_core, (tb.CH_I, tb.CH_J, tb.CH_N), h),
            ('2a T(p,s)', a, t.as_core, (tb.AS_I, tb.AS_J, tb.AS_N), s),
            ('2b T(p,s)', b_s, t.bs_core, (tb.BS_I, tb.BS_J, tb.BS_N), s),
            ('2c T(p,s)', c_s, t.cs_core, (tb.CS_I, tb.CS_J, tb.CS_N), s)):
        n = int(mask.sum())
        assert n, f'подобласть {nm} не задета сеткой'
        front = t.p_h if 'h)' in nm else t.p_s
        xs, ys = pg[mask], y[mask]
        k = min(n, 400)
        direct = np.array([core(float(xs[j]), float(ys[j]), pw_x, pw_y, *tab)
                           for j in range(k)])
        _check(f'{nm} ветвь ({n} точек)', front(xs, ys)[:k], direct, 1e-13)


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  OK    {name}')
        except AssertionError as e:
            bad += 1
            print(f'  FAIL  {name}: {e}')
    print(f'\n{len(tests) - bad} из {len(tests)} прошло')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())

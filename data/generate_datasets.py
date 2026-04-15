"""
Genereer kunstmatige datasets voor Machine Learning in Process Engineering.
Datasets gebaseerd op typische processen in de farmaceutische en chemische industrie.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)


def generate_batch_reactor():
    """
    Dataset 1: Batch Reactor - Exotherme reactie in een farmaceutische batch reactor.
    Sensoren: temperatuur, druk, roersnelheid, koelwatertemperatuur, concentratie product.
    Doel: Voorspel de eindopbrengst (yield) op basis van procesparameters.
    """
    n = 1000

    # Procesparameters
    temp_setpoint = np.random.uniform(60, 90, n)  # °C
    pressure = 1.5 + 0.02 * (temp_setpoint - 60) + np.random.normal(0, 0.05, n)  # bar
    stir_speed = np.random.uniform(100, 500, n)  # RPM
    coolant_temp = np.random.uniform(5, 25, n)  # °C
    reaction_time = np.random.uniform(30, 180, n)  # minuten
    catalyst_amount = np.random.uniform(0.5, 5.0, n)  # gram
    ph = np.random.uniform(5.0, 9.0, n)

    # Yield als niet-lineaire functie van parameters
    yield_pct = (
        50
        + 0.3 * (temp_setpoint - 60)
        - 0.005 * (temp_setpoint - 75) ** 2
        + 0.02 * stir_speed
        - 0.5 * (coolant_temp - 15) ** 2 / 100
        + 0.1 * reaction_time
        - 0.0005 * reaction_time ** 2
        + 2.0 * catalyst_amount
        - 0.3 * catalyst_amount ** 2
        + 1.5 * np.sin(ph - 7) * 5
        + np.random.normal(0, 2, n)
    )
    yield_pct = np.clip(yield_pct, 0, 100)

    # Kwaliteitslabel
    quality = np.where(yield_pct > 85, "Premium", np.where(yield_pct > 70, "Standaard", "Afgekeurd"))

    df = pd.DataFrame({
        "batch_id": [f"BR-{i:04d}" for i in range(n)],
        "temperatuur_C": np.round(temp_setpoint, 1),
        "druk_bar": np.round(pressure, 2),
        "roersnelheid_RPM": np.round(stir_speed, 0).astype(int),
        "koelwater_temp_C": np.round(coolant_temp, 1),
        "reactietijd_min": np.round(reaction_time, 1),
        "katalysator_g": np.round(catalyst_amount, 2),
        "pH": np.round(ph, 1),
        "opbrengst_pct": np.round(yield_pct, 2),
        "kwaliteit": quality,
    })

    df.to_csv("data/batch_reactor_yield.csv", index=False)
    print(f"batch_reactor_yield.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_distillation_column():
    """
    Dataset 2: Distillatiekolom - Continue scheiding van een binair mengsel.
    Sensoren op meerdere trays, tijdreeks data.
    Doel: Voorspel de zuiverheid van het topproduct. Anomalie-detectie.
    """
    n_hours = 720  # 30 dagen
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n_hours)]

    # Feed condities (langzaam variërend)
    feed_flow = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_hours)) + np.random.normal(0, 2, n_hours)
    feed_composition = 0.5 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, n_hours)) + np.random.normal(0, 0.01, n_hours)
    feed_temp = 80 + 5 * np.sin(np.linspace(0, 8 * np.pi, n_hours)) + np.random.normal(0, 1, n_hours)

    # Kolom parameters
    reflux_ratio = 3.0 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, n_hours)) + np.random.normal(0, 0.1, n_hours)
    reboiler_duty = 500 + 30 * np.sin(np.linspace(0, 5 * np.pi, n_hours)) + np.random.normal(0, 10, n_hours)
    condenser_duty = -400 - 25 * np.sin(np.linspace(0, 5 * np.pi, n_hours)) + np.random.normal(0, 8, n_hours)

    # Tray temperaturen (5 trays)
    tray_temps = {}
    base_temps = [120, 110, 100, 90, 80]
    for i, base in enumerate(base_temps):
        tray_temps[f"tray_{i+1}_temp_C"] = (
            base
            + 2 * np.sin(np.linspace(0, 4 * np.pi, n_hours))
            + 0.05 * (feed_temp - 80)
            + np.random.normal(0, 0.5, n_hours)
        )

    # Topproduct zuiverheid
    purity = (
        0.95
        + 0.01 * (reflux_ratio - 3) / 0.5
        + 0.005 * (reboiler_duty - 500) / 30
        - 0.02 * (feed_composition - 0.5) / 0.05
        + np.random.normal(0, 0.003, n_hours)
    )
    purity = np.clip(purity, 0.85, 0.999)

    # Injecteer anomalieën (flooding, foaming)
    anomaly = np.zeros(n_hours, dtype=int)
    anomaly_periods = [(100, 108), (300, 305), (500, 515), (650, 658)]
    for start, end in anomaly_periods:
        anomaly[start:end] = 1
        purity[start:end] -= np.random.uniform(0.03, 0.08, end - start)
        for key in tray_temps:
            tray_temps[key][start:end] += np.random.uniform(3, 8, end - start)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "feed_flow_kgh": np.round(feed_flow, 1),
        "feed_samenstelling_molfrac": np.round(feed_composition, 3),
        "feed_temp_C": np.round(feed_temp, 1),
        "reflux_ratio": np.round(reflux_ratio, 2),
        "reboiler_duty_kW": np.round(reboiler_duty, 1),
        "condenser_duty_kW": np.round(condenser_duty, 1),
        **{k: np.round(v, 1) for k, v in tray_temps.items()},
        "top_zuiverheid": np.round(purity, 4),
        "anomalie": anomaly,
    })

    df.to_csv("data/distillatiekolom_timeseries.csv", index=False)
    print(f"distillatiekolom_timeseries.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_pharma_tablet_press():
    """
    Dataset 3: Farmaceutische tabletpers - Kwaliteitscontrole van tabletten.
    Sensoren: perskracht, vulgewicht, hardheid, dikte, etc.
    Doel: Classificatie (goedgekeurd/afgekeurd) en regressie (hardheid voorspellen).
    """
    n = 2000

    # Grondstof eigenschappen
    api_particle_size = np.random.normal(50, 10, n)  # µm
    moisture_content = np.random.normal(3.0, 0.5, n)  # %
    blend_uniformity = np.random.normal(98, 2, n)  # %

    # Pers parameters
    compression_force = np.random.normal(15, 2, n)  # kN
    pre_compression = np.random.normal(3, 0.5, n)  # kN
    turret_speed = np.random.uniform(20, 60, n)  # RPM
    fill_depth = np.random.normal(12, 0.5, n)  # mm

    # Tablet eigenschappen (outputs)
    weight = (
        350
        + 5 * (fill_depth - 12)
        + 2 * (moisture_content - 3)
        + np.random.normal(0, 3, n)
    )  # mg

    thickness = (
        4.0
        - 0.05 * (compression_force - 15)
        + 0.1 * (fill_depth - 12)
        + np.random.normal(0, 0.05, n)
    )  # mm

    hardness = (
        80
        + 5 * (compression_force - 15)
        + 2 * (pre_compression - 3)
        - 0.5 * (moisture_content - 3) * 10
        - 0.3 * (turret_speed - 40)
        + 0.2 * (api_particle_size - 50)
        + np.random.normal(0, 5, n)
    )  # N

    friability = (
        0.5
        - 0.02 * (compression_force - 15)
        + 0.05 * (moisture_content - 3)
        + 0.005 * (turret_speed - 40)
        + np.random.normal(0, 0.05, n)
    )  # %
    friability = np.clip(friability, 0.01, 2.0)

    dissolution_30min = (
        85
        + 0.5 * (api_particle_size - 50) / 10
        - 2 * (compression_force - 15)
        + 3 * (moisture_content - 3)
        + np.random.normal(0, 3, n)
    )  # %
    dissolution_30min = np.clip(dissolution_30min, 40, 100)

    # Goedkeuring op basis van specificaties
    approved = (
        (weight > 332.5) & (weight < 367.5) &  # ±5%
        (hardness > 50) & (hardness < 120) &
        (friability < 1.0) &
        (dissolution_30min > 75) &
        (thickness > 3.5) & (thickness < 4.5)
    ).astype(int)

    df = pd.DataFrame({
        "tablet_id": [f"TAB-{i:05d}" for i in range(n)],
        "deeltjesgrootte_um": np.round(api_particle_size, 1),
        "vochtgehalte_pct": np.round(moisture_content, 2),
        "menguniformiteit_pct": np.round(blend_uniformity, 1),
        "perskracht_kN": np.round(compression_force, 1),
        "voorcompressie_kN": np.round(pre_compression, 1),
        "draaisnelheid_RPM": np.round(turret_speed, 0).astype(int),
        "vuldiepte_mm": np.round(fill_depth, 2),
        "gewicht_mg": np.round(weight, 1),
        "dikte_mm": np.round(thickness, 2),
        "hardheid_N": np.round(hardness, 1),
        "brosheid_pct": np.round(friability, 3),
        "dissolutie_30min_pct": np.round(dissolution_30min, 1),
        "goedgekeurd": approved,
    })

    df.to_csv("data/tabletpers_kwaliteit.csv", index=False)
    print(f"tabletpers_kwaliteit.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_cstr_experiment():
    """
    Dataset 4: Continuous Stirred Tank Reactor (CSTR) - Meervoudige steady states.
    Sensoren: inlet/outlet temperatuur, concentratie, flowrates.
    Doel: Regressie, clustering van operatieregimes.
    """
    n = 1500

    # Drie operatieregimes
    regime = np.random.choice(["laag", "midden", "hoog"], n, p=[0.3, 0.4, 0.3])

    regime_params = {
        "laag": {"T_out": 320, "C_out": 0.8, "conv": 20},
        "midden": {"T_out": 370, "C_out": 0.4, "conv": 60},
        "hoog": {"T_out": 420, "C_out": 0.1, "conv": 90},
    }

    inlet_temp = np.random.normal(300, 5, n)  # K
    inlet_conc = np.random.normal(1.0, 0.05, n)  # mol/L
    inlet_flow = np.random.uniform(1.0, 5.0, n)  # L/min
    coolant_flow = np.random.uniform(0.5, 3.0, n)  # L/min
    volume = np.random.choice([5, 10, 20], n)  # L

    outlet_temp = np.array([regime_params[r]["T_out"] for r in regime], dtype=float)
    outlet_temp += 0.1 * (inlet_temp - 300) - 2 * (coolant_flow - 1.5) + np.random.normal(0, 3, n)

    outlet_conc = np.array([regime_params[r]["C_out"] for r in regime], dtype=float)
    outlet_conc += 0.05 * (inlet_conc - 1.0) + 0.02 * (inlet_flow - 3) + np.random.normal(0, 0.03, n)
    outlet_conc = np.clip(outlet_conc, 0.01, 1.0)

    conversion = np.array([regime_params[r]["conv"] for r in regime], dtype=float)
    conversion += np.random.normal(0, 3, n)
    conversion = np.clip(conversion, 0, 100)

    residence_time = volume / inlet_flow  # min

    df = pd.DataFrame({
        "experiment_id": [f"CSTR-{i:04d}" for i in range(n)],
        "inlet_temp_K": np.round(inlet_temp, 1),
        "inlet_conc_molL": np.round(inlet_conc, 3),
        "inlet_flow_Lmin": np.round(inlet_flow, 2),
        "koelwater_flow_Lmin": np.round(coolant_flow, 2),
        "reactor_volume_L": volume,
        "verblijftijd_min": np.round(residence_time, 2),
        "outlet_temp_K": np.round(outlet_temp, 1),
        "outlet_conc_molL": np.round(outlet_conc, 3),
        "conversie_pct": np.round(conversion, 1),
        "operatieregime": regime,
    })

    df.to_csv("data/cstr_experiment.csv", index=False)
    print(f"cstr_experiment.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_sensor_drift():
    """
    Dataset 5: Sensor Drift & Predictive Maintenance - pH en temperatuursensoren
    met geleidelijke degradatie over tijd.
    Doel: Voorspel wanneer een sensor gekalibreerd moet worden.
    """
    n_sensors = 10
    n_days = 180
    n_measurements = 24  # per dag

    rows = []
    for sensor_id in range(n_sensors):
        # Elke sensor heeft een andere drift rate
        drift_rate = np.random.uniform(0.001, 0.01)
        noise_level = np.random.uniform(0.02, 0.1)
        failure_day = np.random.randint(90, 180)

        true_value = 7.0  # pH

        for day in range(n_days):
            for hour in range(n_measurements):
                t = day * n_measurements + hour

                # Geleidelijke drift
                drift = drift_rate * day

                # Ruis neemt toe naarmate sensor degradeert
                current_noise = noise_level * (1 + 0.01 * day)

                # Meetwaarde
                measured = true_value + drift + np.random.normal(0, current_noise)

                # Na failure day: grote afwijkingen
                if day > failure_day:
                    measured += np.random.uniform(0.5, 2.0)

                # Sensor status
                error = abs(measured - true_value)
                if error > 1.0:
                    status = "defect"
                elif error > 0.5:
                    status = "kalibratie_nodig"
                else:
                    status = "ok"

                rows.append({
                    "sensor_id": f"pH-{sensor_id:02d}",
                    "dag": day,
                    "uur": hour,
                    "werkelijke_pH": true_value,
                    "gemeten_pH": round(measured, 3),
                    "afwijking": round(measured - true_value, 3),
                    "ruis_std": round(current_noise, 4),
                    "status": status,
                })

    df = pd.DataFrame(rows)
    df.to_csv("data/sensor_drift_maintenance.csv", index=False)
    print(f"sensor_drift_maintenance.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_crystallization():
    """
    Dataset 6: Kristallisatieproces - Farmaceutische API kristallisatie.
    Sensoren: temperatuurprofiel, verzadiging, kristalgrootte.
    Doel: Voorspel kristalgrootteverdeling, optimaliseer koelprofiel.
    """
    n = 800

    # Procesparameters
    initial_temp = np.random.uniform(60, 80, n)  # °C
    final_temp = np.random.uniform(5, 25, n)  # °C
    cooling_rate = np.random.uniform(0.1, 2.0, n)  # °C/min
    initial_concentration = np.random.uniform(100, 300, n)  # g/L
    solvent = np.random.choice(["water", "ethanol", "methanol", "aceton"], n)
    seed_amount = np.random.uniform(0, 5, n)  # g
    stir_speed = np.random.uniform(50, 300, n)  # RPM
    anti_solvent_rate = np.random.uniform(0, 10, n)  # mL/min

    # Solvent effect
    solvent_factor = {"water": 1.0, "ethanol": 1.2, "methanol": 1.1, "aceton": 0.9}
    sf = np.array([solvent_factor[s] for s in solvent])

    # Gemiddelde kristalgrootte (niet-lineair model)
    mean_crystal_size = (
        50 * sf
        + 20 * np.log1p(seed_amount)
        - 30 * cooling_rate
        + 0.5 * (initial_temp - final_temp)
        + 0.1 * stir_speed
        - 5 * anti_solvent_rate / (anti_solvent_rate + 1)
        + np.random.normal(0, 8, n)
    )  # µm
    mean_crystal_size = np.clip(mean_crystal_size, 5, 500)

    # Kristalgrootte spreiding (CV%)
    size_cv = (
        30
        + 10 * cooling_rate
        - 5 * np.log1p(seed_amount)
        - 0.05 * stir_speed
        + np.random.normal(0, 3, n)
    )
    size_cv = np.clip(size_cv, 5, 60)

    # Opbrengst
    yield_pct = (
        70
        + 0.1 * (initial_concentration - 200)
        + 0.2 * (initial_temp - final_temp)
        - 5 * cooling_rate
        + np.random.normal(0, 3, n)
    )
    yield_pct = np.clip(yield_pct, 30, 99)

    # Polymorf (kristalvorm)
    polymorph_prob = 1 / (1 + np.exp(-(cooling_rate - 1.0) * 3))
    polymorph = np.where(np.random.random(n) < polymorph_prob, "Form_B", "Form_A")

    df = pd.DataFrame({
        "experiment_id": [f"CRYST-{i:04d}" for i in range(n)],
        "start_temp_C": np.round(initial_temp, 1),
        "eind_temp_C": np.round(final_temp, 1),
        "koelsnelheid_C_min": np.round(cooling_rate, 2),
        "concentratie_gL": np.round(initial_concentration, 1),
        "oplosmiddel": solvent,
        "zaadkristallen_g": np.round(seed_amount, 2),
        "roersnelheid_RPM": np.round(stir_speed, 0).astype(int),
        "anti_solvent_rate_mLmin": np.round(anti_solvent_rate, 1),
        "gem_kristalgrootte_um": np.round(mean_crystal_size, 1),
        "kristalgrootte_CV_pct": np.round(size_cv, 1),
        "opbrengst_pct": np.round(yield_pct, 1),
        "polymorf": polymorph,
    })

    df.to_csv("data/kristallisatie_proces.csv", index=False)
    print(f"kristallisatie_proces.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_wwtp():
    """
    Dataset 7: Afvalwaterbehandeling (WWTP) - Chemische/farma industrie.
    Sensoren: COD, pH, TSS, debiet, beluchting.
    Doel: Voorspel effluent kwaliteit, optimaliseer beluchting.
    """
    n_hours = 2160  # 90 dagen
    timestamps = [datetime(2025, 3, 1) + timedelta(hours=i) for i in range(n_hours)]

    # Dag/nacht cyclus
    hour_of_day = np.array([t.hour for t in timestamps])
    day_factor = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)

    # Influent
    influent_flow = 500 * day_factor + np.random.normal(0, 30, n_hours)  # m³/h
    influent_cod = 800 * day_factor + np.random.normal(0, 50, n_hours)  # mg/L
    influent_ph = 7.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 7)) + np.random.normal(0, 0.2, n_hours)
    influent_tss = 300 * day_factor + np.random.normal(0, 20, n_hours)  # mg/L
    influent_nh4 = 40 * day_factor + np.random.normal(0, 5, n_hours)  # mg/L

    # Procesparameters
    aeration_rate = 200 + 50 * day_factor + np.random.normal(0, 10, n_hours)  # m³/h
    sludge_recycle = np.random.uniform(0.3, 0.8, n_hours)  # ratio
    srt = np.random.uniform(10, 25, n_hours)  # dagen (sludge retention time)
    do_level = 1.5 + 0.5 * (aeration_rate - 250) / 50 + np.random.normal(0, 0.2, n_hours)  # mg/L

    # Effluent kwaliteit
    cod_removal = 0.85 + 0.05 * (do_level - 2) / 0.5 + 0.02 * (srt - 15) / 5 + np.random.normal(0, 0.02, n_hours)
    cod_removal = np.clip(cod_removal, 0.6, 0.98)

    effluent_cod = influent_cod * (1 - cod_removal)
    effluent_ph = 7.0 + 0.3 * (influent_ph - 7.5) + np.random.normal(0, 0.1, n_hours)
    effluent_tss = influent_tss * np.random.uniform(0.02, 0.1, n_hours)
    effluent_nh4 = influent_nh4 * (0.1 + 0.05 * np.exp(-0.3 * (do_level - 1))) + np.random.normal(0, 1, n_hours)
    effluent_nh4 = np.clip(effluent_nh4, 0.1, 30)

    # Lozingsnorm overschrijding
    norm_overtreding = (
        (effluent_cod > 125) | (effluent_nh4 > 10) | (effluent_tss > 35)
    ).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "influent_debiet_m3h": np.round(influent_flow, 1),
        "influent_COD_mgL": np.round(influent_cod, 1),
        "influent_pH": np.round(influent_ph, 1),
        "influent_TSS_mgL": np.round(influent_tss, 1),
        "influent_NH4_mgL": np.round(influent_nh4, 1),
        "beluchting_m3h": np.round(aeration_rate, 1),
        "slibrecirculatie_ratio": np.round(sludge_recycle, 2),
        "slibverblijftijd_dagen": np.round(srt, 1),
        "opgeloste_O2_mgL": np.round(do_level, 2),
        "effluent_COD_mgL": np.round(effluent_cod, 1),
        "effluent_pH": np.round(effluent_ph, 1),
        "effluent_TSS_mgL": np.round(effluent_tss, 1),
        "effluent_NH4_mgL": np.round(effluent_nh4, 1),
        "lozingsnorm_overtreding": norm_overtreding,
    })

    df.to_csv("data/afvalwater_behandeling.csv", index=False)
    print(f"afvalwater_behandeling.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_spc_controlchart():
    """
    Dataset 8: Statistical Process Control (SPC) - Controlekaartendata voor een
    continu chemisch proces. Bevat normale variatie, trends, shifts en regelovertredingen.
    Doel: Detectie van out-of-control situaties, patroonherkenning op controlekaarten.
    """
    n_samples = 2000

    # Basisproces: vulgewicht van een verpakkingslijn
    target = 500.0  # gram
    sigma = 2.0

    measurement = np.random.normal(target, sigma, n_samples)
    pattern = np.full(n_samples, "normaal", dtype=object)

    # Patroon 1: Geleidelijke trend (drift omhoog) - samples 200-300
    trend = np.linspace(0, 8, 100)
    measurement[200:300] += trend
    pattern[200:300] = "trend_omhoog"

    # Patroon 2: Plotse shift - samples 500-580
    measurement[500:580] += 5.0
    pattern[500:580] = "shift_omhoog"

    # Patroon 3: Verhoogde variatie - samples 800-900
    measurement[800:900] = np.random.normal(target, sigma * 2.5, 100)
    pattern[800:900] = "verhoogde_variatie"

    # Patroon 4: Cyclisch patroon - samples 1100-1250
    cycle = 3 * np.sin(np.linspace(0, 6 * np.pi, 150))
    measurement[1100:1250] += cycle
    pattern[1100:1250] = "cyclisch"

    # Patroon 5: Geleidelijke trend omlaag - samples 1500-1600
    trend_down = np.linspace(0, -6, 100)
    measurement[1500:1600] += trend_down
    pattern[1500:1600] = "trend_omlaag"

    # Patroon 6: Stratificatie (te weinig variatie) - samples 1750-1850
    measurement[1750:1850] = np.random.normal(target, sigma * 0.3, 100)
    pattern[1750:1850] = "stratificatie"

    # Bereken SPC statistieken
    ucl = target + 3 * sigma
    lcl = target - 3 * sigma
    buiten_limieten = ((measurement > ucl) | (measurement < lcl)).astype(int)

    # Subgroepen van 5
    subgroup = np.repeat(np.arange(n_samples // 5), 5)[:n_samples]

    df = pd.DataFrame({
        "sample_nr": np.arange(1, n_samples + 1),
        "subgroep": subgroup,
        "meetwaarde_g": np.round(measurement, 2),
        "doel_g": target,
        "UCL_g": ucl,
        "LCL_g": lcl,
        "buiten_limieten": buiten_limieten,
        "patroon": pattern,
    })

    df.to_csv("data/spc_controlekaart.csv", index=False)
    print(f"spc_controlekaart.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_incoming_qc():
    """
    Dataset 9: Ingangscontrole grondstoffen - Kwaliteitscontrole van binnenkomende
    grondstoffen voor farmaceutische productie (GMP-omgeving).
    Doel: Classificatie (goedkeuren/afkeuren/hertest), anomalie-detectie op leveranciers.
    """
    n = 1200

    leverancier = np.random.choice(
        ["Leverancier_A", "Leverancier_B", "Leverancier_C", "Leverancier_D"],
        n, p=[0.35, 0.30, 0.20, 0.15]
    )
    grondstof = np.random.choice(
        ["Paracetamol_API", "Lactose", "Magnesiumstearaat", "Cellulose_MCC"],
        n, p=[0.30, 0.30, 0.20, 0.20]
    )

    # Leverancier-specifieke kwaliteitsverschillen
    lev_offset = {"Leverancier_A": 0, "Leverancier_B": 0.5, "Leverancier_C": -0.3, "Leverancier_D": 1.2}
    offset = np.array([lev_offset[l] for l in leverancier])

    # Analytische testen
    zuiverheid = 99.5 + offset * 0.1 + np.random.normal(0, 0.3, n)  # %
    zuiverheid = np.clip(zuiverheid, 95, 100)

    vochtgehalte = 2.0 - offset * 0.1 + np.random.normal(0, 0.3, n)  # %
    vochtgehalte = np.clip(vochtgehalte, 0.1, 5.0)

    deeltjesgrootte_d50 = 80 + offset * 5 + np.random.normal(0, 10, n)  # µm
    deeltjesgrootte_d90 = deeltjesgrootte_d50 * 1.8 + np.random.normal(0, 5, n)

    bulkdichtheid = 0.45 + offset * 0.02 + np.random.normal(0, 0.03, n)  # g/mL
    tapdichtheid = bulkdichtheid * 1.25 + np.random.normal(0, 0.02, n)
    carr_index = 100 * (1 - bulkdichtheid / tapdichtheid)  # %

    zware_metalen = 5 + np.abs(offset) * 2 + np.random.exponential(2, n)  # ppm
    zware_metalen = np.clip(zware_metalen, 0.1, 50)

    microbieel = np.random.choice([0, 10, 50, 100, 500, 1000], n, p=[0.4, 0.25, 0.15, 0.1, 0.07, 0.03])  # CFU/g

    # Besluit op basis van specs
    besluit = np.full(n, "goedgekeurd", dtype=object)
    besluit[(zuiverheid < 98.0) | (zware_metalen > 20) | (microbieel > 500)] = "afgekeurd"
    besluit[(zuiverheid < 99.0) & (besluit != "afgekeurd")] = "hertest"
    besluit[(vochtgehalte > 4.0) & (besluit != "afgekeurd")] = "hertest"

    # Lotnummers
    lot_dates = [datetime(2024, 6, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n)]

    df = pd.DataFrame({
        "lot_id": [f"LOT-{i:05d}" for i in range(n)],
        "datum": [d.strftime("%Y-%m-%d") for d in lot_dates],
        "leverancier": leverancier,
        "grondstof": grondstof,
        "zuiverheid_pct": np.round(zuiverheid, 2),
        "vochtgehalte_pct": np.round(vochtgehalte, 2),
        "deeltjesgrootte_d50_um": np.round(deeltjesgrootte_d50, 1),
        "deeltjesgrootte_d90_um": np.round(deeltjesgrootte_d90, 1),
        "bulkdichtheid_gmL": np.round(bulkdichtheid, 3),
        "tapdichtheid_gmL": np.round(tapdichtheid, 3),
        "carr_index_pct": np.round(carr_index, 1),
        "zware_metalen_ppm": np.round(zware_metalen, 1),
        "microbieel_CFUg": microbieel,
        "besluit": besluit,
    })

    df.to_csv("data/ingangscontrole_grondstoffen.csv", index=False)
    print(f"ingangscontrole_grondstoffen.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_fermentation():
    """
    Dataset 10: Fermentatieproces - Biofarmaceutische productie (fed-batch).
    Tijdreeksdata van een bioreaktor met voedingsstrategieën.
    Doel: Voorspel productconcentratie, optimaliseer voedingsstrategie.
    """
    n_batches = 50
    n_timepoints = 100  # metingen per batch (elke 2 uur, ~8 dagen)

    rows = []
    for batch in range(n_batches):
        # Batch-specifieke parameters
        inoculum_density = np.random.uniform(0.5, 2.0)  # OD600
        initial_glucose = np.random.uniform(15, 25)  # g/L
        temp_setpoint = np.random.choice([30, 33, 37])  # °C
        ph_setpoint = np.random.choice([6.8, 7.0, 7.2])
        feed_strategy = np.random.choice(["constant", "exponentieel", "DO_gestuurd"])

        # Groeifase
        mu_max = 0.3 + 0.02 * (temp_setpoint - 33) + np.random.normal(0, 0.02)
        lag_time = np.random.uniform(2, 8)  # uur

        for t_idx in range(n_timepoints):
            t = t_idx * 2  # uur

            # Biomassa (logistische groei)
            biomassa = inoculum_density * 50 / (
                inoculum_density + (50 - inoculum_density) * np.exp(-mu_max * max(0, t - lag_time) / 10)
            )
            biomassa += np.random.normal(0, 0.3)
            biomassa = max(0.1, biomassa)

            # Glucose
            glucose = max(0, initial_glucose - biomassa * 0.4 + np.random.normal(0, 0.5))

            # Voeding
            if feed_strategy == "constant":
                feed_rate = 2.0 if t > 20 else 0
            elif feed_strategy == "exponentieel":
                feed_rate = 0.5 * np.exp(0.03 * max(0, t - 20)) if t > 20 else 0
                feed_rate = min(feed_rate, 10)
            else:  # DO_gestuurd
                feed_rate = 3.0 if (t > 20 and glucose < 5) else 0

            # Opgeloste zuurstof
            do = 80 - 2 * biomassa + 5 * (1 if feed_rate == 0 else 0) + np.random.normal(0, 3)
            do = np.clip(do, 5, 100)

            # Product (recombinant eiwit)
            product = 0.05 * biomassa * max(0, t - lag_time) / 50 * (1 + 0.1 * (ph_setpoint - 7))
            product += np.random.normal(0, 0.02)
            product = max(0, product)

            # pH drift
            ph_actual = ph_setpoint + 0.1 * np.sin(t / 20) + np.random.normal(0, 0.05)

            # Temperatuur
            temp_actual = temp_setpoint + np.random.normal(0, 0.2)

            rows.append({
                "batch_id": f"FERM-{batch:03d}",
                "tijdstip_uur": t,
                "temp_setpoint_C": temp_setpoint,
                "temp_actueel_C": round(temp_actual, 1),
                "pH_setpoint": ph_setpoint,
                "pH_actueel": round(ph_actual, 2),
                "biomassa_gL": round(biomassa, 2),
                "glucose_gL": round(glucose, 2),
                "voedingssnelheid_gLh": round(feed_rate, 2),
                "voedingsstrategie": feed_strategy,
                "opgeloste_O2_pct": round(do, 1),
                "product_gL": round(product, 3),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/fermentatie_bioreaktor.csv", index=False)
    print(f"fermentatie_bioreaktor.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_hplc_stability():
    """
    Dataset 11: HPLC Stabiliteitsonderzoek - Farma product stabiliteit over tijd
    bij verschillende opslagcondities (ICH richtlijnen).
    Doel: Voorspel houdbaarheid, regressie op degradatiekinetiek.
    """
    n_batches = 30
    conditions = [
        ("25C_60RH", 25, 60),    # Long-term
        ("30C_65RH", 30, 65),    # Intermediate
        ("40C_75RH", 40, 75),    # Accelerated
    ]
    timepoints_months = [0, 1, 2, 3, 6, 9, 12, 18, 24, 36]

    rows = []
    for batch in range(n_batches):
        initial_assay = np.random.normal(100, 0.5)  # %
        initial_impurity_a = np.random.uniform(0.05, 0.15)  # %
        initial_impurity_b = np.random.uniform(0.01, 0.05)  # %
        initial_water = np.random.normal(2.0, 0.3)  # %
        formulation = np.random.choice(["tablet", "capsule", "suspensie"])

        form_factor = {"tablet": 1.0, "capsule": 1.3, "suspensie": 1.8}[formulation]

        for cond_name, temp, rh in conditions:
            # Arrhenius-achtige degradatie
            k = 0.001 * form_factor * np.exp(0.05 * (temp - 25))

            for month in timepoints_months:
                # Gehalte daalt
                assay = initial_assay * np.exp(-k * month) + np.random.normal(0, 0.2)

                # Onzuiverheden stijgen
                imp_a = initial_impurity_a + k * month * 5 + np.random.normal(0, 0.02)
                imp_b = initial_impurity_b + k * month * 2 + np.random.normal(0, 0.01)
                total_imp = imp_a + imp_b + np.random.uniform(0, 0.1)

                # Water content
                water = initial_water + 0.02 * rh / 60 * month + np.random.normal(0, 0.1)

                # Dissolution (tabletten/capsules)
                dissolution = 95 - 0.5 * month * k * 10 + np.random.normal(0, 2) if formulation != "suspensie" else np.nan

                # Uiterlijk score (1-5, 5=perfect)
                appearance = max(1, 5 - 0.1 * month * k * 10 + np.random.normal(0, 0.3))

                # OOS (out of specification)
                oos = int(assay < 95 or total_imp > 2.0 or (dissolution is not np.nan and dissolution < 75))

                rows.append({
                    "batch_id": f"STAB-{batch:03d}",
                    "formulering": formulation,
                    "opslagconditie": cond_name,
                    "temperatuur_C": temp,
                    "rel_vochtigheid_pct": rh,
                    "tijdstip_maanden": month,
                    "gehalte_pct": round(assay, 2),
                    "onzuiverheid_A_pct": round(imp_a, 3),
                    "onzuiverheid_B_pct": round(imp_b, 3),
                    "totaal_onzuiverheden_pct": round(total_imp, 3),
                    "watergehalte_pct": round(water, 2),
                    "dissolutie_pct": round(dissolution, 1) if not np.isnan(dissolution) else None,
                    "uiterlijk_score": round(min(5, appearance), 1),
                    "out_of_spec": oos,
                })

    df = pd.DataFrame(rows)
    df.to_csv("data/hplc_stabiliteit.csv", index=False)
    print(f"hplc_stabiliteit.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_heat_exchanger():
    """
    Dataset 12: Warmtewisselaar fouling monitoring - Continue procesmonitoring
    van een shell-and-tube warmtewisselaar in een chemische plant.
    Doel: Voorspel fouling factor, plan onderhoud (predictive maintenance).
    """
    n_days = 365
    n_per_day = 24

    rows = []
    fouling_factor = 0.0  # starts clean
    last_cleaning = 0

    for day in range(n_days):
        # Schoonmaakactie elke ~90 dagen
        if day - last_cleaning > np.random.randint(80, 100) and day > 10:
            fouling_factor = np.random.uniform(0, 0.05)  # niet perfect schoon
            last_cleaning = day

        for hour in range(n_per_day):
            # Procesomstandigheden
            hot_inlet_temp = 150 + 10 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 2)  # °C
            hot_flow = 50 + np.random.normal(0, 3)  # m³/h
            cold_inlet_temp = 20 + 8 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 1)  # °C
            cold_flow = 60 + np.random.normal(0, 4)  # m³/h

            # Fouling groeit over tijd (niet-lineair)
            days_since_clean = day - last_cleaning
            fouling_factor = 0.0005 * days_since_clean ** 1.3 + np.random.normal(0, 0.002)
            fouling_factor = max(0, fouling_factor)

            # Overall heat transfer coefficient daalt met fouling
            u_clean = 500  # W/(m²·K)
            u_actual = u_clean / (1 + fouling_factor * u_clean)

            # Outlet temperaturen
            hot_outlet_temp = hot_inlet_temp - (hot_inlet_temp - cold_inlet_temp) * 0.6 / (1 + fouling_factor * 100)
            hot_outlet_temp += np.random.normal(0, 0.5)

            cold_outlet_temp = cold_inlet_temp + (hot_inlet_temp - cold_inlet_temp) * 0.5 / (1 + fouling_factor * 100)
            cold_outlet_temp += np.random.normal(0, 0.5)

            # Drukval stijgt met fouling
            dp_hot = 0.5 + 0.3 * fouling_factor * 100 + np.random.normal(0, 0.02)  # bar
            dp_cold = 0.4 + 0.2 * fouling_factor * 100 + np.random.normal(0, 0.02)  # bar

            # Effectiveness
            effectiveness = (hot_inlet_temp - hot_outlet_temp) / (hot_inlet_temp - cold_inlet_temp)

            # Onderhoud nodig?
            onderhoud = "gepland" if fouling_factor > 0.003 else ("waarschuwing" if fouling_factor > 0.002 else "ok")

            rows.append({
                "datum": (datetime(2025, 1, 1) + timedelta(days=day, hours=hour)).strftime("%Y-%m-%d %H:%M"),
                "dag_sinds_reiniging": days_since_clean,
                "hot_inlet_C": round(hot_inlet_temp, 1),
                "hot_outlet_C": round(hot_outlet_temp, 1),
                "hot_flow_m3h": round(hot_flow, 1),
                "cold_inlet_C": round(cold_inlet_temp, 1),
                "cold_outlet_C": round(cold_outlet_temp, 1),
                "cold_flow_m3h": round(cold_flow, 1),
                "drukval_hot_bar": round(dp_hot, 3),
                "drukval_cold_bar": round(dp_cold, 3),
                "U_actueel_Wm2K": round(u_actual, 1),
                "effectiviteit": round(effectiveness, 3),
                "fouling_factor": round(fouling_factor, 5),
                "onderhoudsstatus": onderhoud,
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/warmtewisselaar_fouling.csv", index=False)
    print(f"warmtewisselaar_fouling.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_continuous_polymerization():
    """
    Dataset 13: Continue polymerisatiereactor - Productie van polyethyleen.
    Hoge-frequentie sensordata (elke minuut) over 14 dagen.
    Meerdere grade-transities, procesverstoringen, en PID-regelaaracties.
    Doel: Soft sensor (viscositeit), grade-transitie optimalisatie, procesbewaking.
    """
    n_minutes = 14 * 24 * 60  # 14 dagen, per minuut
    t = np.arange(n_minutes)
    timestamps = [datetime(2025, 1, 1) + timedelta(minutes=int(m)) for m in t]

    # --- Grade schedule (4 grades, met transities) ---
    grade_schedule = np.zeros(n_minutes, dtype=int)
    grade_names = ["HDPE_A", "HDPE_B", "LDPE_C", "LLDPE_D"]
    grade_props = {
        "HDPE_A":  {"temp": 180, "pressure": 25, "cat_flow": 2.0, "monomer_flow": 100, "target_mfi": 2.0},
        "HDPE_B":  {"temp": 190, "pressure": 28, "cat_flow": 2.5, "monomer_flow": 110, "target_mfi": 5.0},
        "LDPE_C":  {"temp": 220, "pressure": 200, "cat_flow": 1.5, "monomer_flow": 80, "target_mfi": 8.0},
        "LLDPE_D": {"temp": 200, "pressure": 30, "cat_flow": 1.8, "monomer_flow": 95, "target_mfi": 1.0},
    }

    # Grade wissels op specifieke tijdstippen
    transitions = [0, 4800, 8500, 12000, 16000]  # minuten
    grade_order = [0, 1, 2, 3, 0]
    for i in range(len(transitions)):
        start = transitions[i]
        end = transitions[i + 1] if i + 1 < len(transitions) else n_minutes
        grade_schedule[start:end] = grade_order[i]

    # Smooth setpoints met transitietijd (~60 min)
    def smooth_setpoint(schedule, prop_key):
        raw = np.array([grade_props[grade_names[g]][prop_key] for g in schedule])
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(raw.astype(float), size=60)

    temp_sp = smooth_setpoint(grade_schedule, "temp")
    press_sp = smooth_setpoint(grade_schedule, "pressure")
    cat_sp = smooth_setpoint(grade_schedule, "cat_flow")
    mono_sp = smooth_setpoint(grade_schedule, "monomer_flow")
    target_mfi = smooth_setpoint(grade_schedule, "target_mfi")

    # --- Actuele waarden met PID-gedrag (traag volgend) ---
    temp_actual = np.zeros(n_minutes)
    press_actual = np.zeros(n_minutes)
    temp_actual[0] = temp_sp[0]
    press_actual[0] = press_sp[0]

    for i in range(1, n_minutes):
        temp_actual[i] = temp_actual[i-1] + 0.05 * (temp_sp[i] - temp_actual[i-1]) + np.random.normal(0, 0.15)
        press_actual[i] = press_actual[i-1] + 0.08 * (press_sp[i] - press_actual[i-1]) + np.random.normal(0, 0.1)

    cat_flow = cat_sp + np.random.normal(0, 0.02, n_minutes)
    monomer_flow = mono_sp + np.random.normal(0, 0.5, n_minutes)
    h2_flow = 0.5 + 0.3 * (target_mfi - 2) / 6 + np.random.normal(0, 0.01, n_minutes)
    comonomer_flow = np.where(
        np.array([grade_names[g] for g in grade_schedule]) == "LLDPE_D",
        5.0 + np.random.normal(0, 0.1, n_minutes),
        0.1 + np.random.normal(0, 0.01, n_minutes)
    )

    # Koelwater
    coolant_inlet = 25 + 3 * np.sin(2 * np.pi * t / (24 * 60)) + np.random.normal(0, 0.3, n_minutes)
    coolant_outlet = coolant_inlet + 15 + 0.05 * (temp_actual - 190) + np.random.normal(0, 0.5, n_minutes)
    coolant_flow = 50 + np.random.normal(0, 1, n_minutes)

    # --- Productkwaliteit (soft sensor targets) ---
    melt_flow_index = (
        target_mfi
        + 0.1 * (temp_actual - temp_sp)
        + 0.05 * (h2_flow - 0.5) * 10
        - 0.02 * (cat_flow - cat_sp) * 5
        + np.random.normal(0, 0.15, n_minutes)
    )
    melt_flow_index = np.clip(melt_flow_index, 0.1, 20)

    # Viscositeit (omgekeerd gerelateerd aan MFI)
    viscosity = 5000 / melt_flow_index + np.random.normal(0, 30, n_minutes)

    density = (
        0.950
        - 0.005 * comonomer_flow / 5
        + 0.001 * (press_actual - 30) / 10
        + np.random.normal(0, 0.001, n_minutes)
    )

    # --- Procesverstoringen injecteren ---
    # Verstoring 1: Katalysator puls (min 3000-3020)
    cat_flow[3000:3020] *= 1.8
    melt_flow_index[3010:3060] += np.linspace(0, 2, 50)
    temp_actual[3005:3050] += np.linspace(0, 5, 45)

    # Verstoring 2: Koelwateruitval (min 7200-7230)
    coolant_flow[7200:7230] *= 0.2
    temp_actual[7210:7280] += np.linspace(0, 12, 70) * np.exp(-np.linspace(0, 3, 70))

    # Verstoring 3: Monomer verontreiniging (min 11000-11200)
    monomer_flow[11000:11200] += np.random.normal(3, 1, 200)
    melt_flow_index[11050:11250] += np.random.normal(0.5, 0.3, 200)

    # Verstoring 4: Sensor spike (min 15000-15005)
    temp_spike = temp_actual.copy()
    temp_spike[15000:15005] += 50  # sensor fout, niet echt

    # Verstoringslabels
    verstoring = np.full(n_minutes, "normaal", dtype=object)
    verstoring[3000:3060] = "katalysator_puls"
    verstoring[7200:7280] = "koelwater_uitval"
    verstoring[11000:11250] = "monomer_verontreiniging"
    verstoring[15000:15005] = "sensor_fout"

    # Transitie labels
    in_transitie = np.zeros(n_minutes, dtype=int)
    for tr in transitions[1:]:
        in_transitie[tr:tr+120] = 1  # 2 uur transitie

    # Energieverbruik
    vermogen_kW = (
        50
        + 0.3 * monomer_flow
        + 0.1 * (temp_actual - 180)
        + 0.05 * press_actual
        + np.random.normal(0, 2, n_minutes)
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "grade": [grade_names[g] for g in grade_schedule],
        "in_transitie": in_transitie,
        "temp_setpoint_C": np.round(temp_sp, 1),
        "temp_actueel_C": np.round(temp_actual, 2),
        "temp_sensor_C": np.round(temp_spike, 2),
        "druk_setpoint_bar": np.round(press_sp, 1),
        "druk_actueel_bar": np.round(press_actual, 2),
        "katalysator_flow_kgh": np.round(cat_flow, 3),
        "monomeer_flow_kgh": np.round(monomer_flow, 1),
        "waterstof_flow_kgh": np.round(h2_flow, 3),
        "comonomeer_flow_kgh": np.round(comonomer_flow, 2),
        "koelwater_inlet_C": np.round(coolant_inlet, 1),
        "koelwater_outlet_C": np.round(coolant_outlet, 1),
        "koelwater_flow_m3h": np.round(coolant_flow, 1),
        "melt_flow_index_g10min": np.round(melt_flow_index, 2),
        "viscositeit_Pas": np.round(viscosity, 1),
        "dichtheid_gcm3": np.round(density, 4),
        "vermogen_kW": np.round(vermogen_kW, 1),
        "verstoring": verstoring,
    })

    df.to_csv("data/continue_polymerisatie.csv", index=False)
    print(f"continue_polymerisatie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_continuous_drying():
    """
    Dataset 14: Continue wervelbeddroger - Farmaceutische granulaatdroging.
    Sensordata elke 30 seconden over 7 dagen. Meerdere batches achter elkaar.
    Doel: Voorspel vochtgehalte (soft sensor), detecteer eindpunt, energieoptimalisatie.
    """
    n_per_batch = 600  # 30 sec interval, ~5 uur per batch
    n_batches = 30
    n_total = n_per_batch * n_batches

    rows = []
    base_time = datetime(2025, 2, 1)
    global_idx = 0

    for batch in range(n_batches):
        # Batch-specifieke variatie
        initial_moisture = np.random.uniform(25, 35)  # %
        target_moisture = np.random.uniform(1.5, 3.0)  # %
        inlet_air_temp_sp = np.random.choice([55, 60, 65, 70])  # °C
        airflow_sp = np.random.uniform(800, 1200)  # m³/h
        bed_mass = np.random.uniform(80, 120)  # kg

        # Droogconstante (afhankelijk van instellingen)
        k_dry = 0.004 * (inlet_air_temp_sp / 60) * (airflow_sp / 1000)

        for i in range(n_per_batch):
            t_sec = i * 30
            t_min = t_sec / 60
            timestamp = base_time + timedelta(seconds=global_idx * 30)

            # Vochtgehalte: exponentieel dalend
            moisture = target_moisture + (initial_moisture - target_moisture) * np.exp(-k_dry * t_min)
            moisture += np.random.normal(0, 0.3)
            moisture = max(0.5, moisture)

            # Inlaatlucht temperatuur (PID-geregeld)
            inlet_air_temp = inlet_air_temp_sp + np.random.normal(0, 0.5)

            # Uitlaatlucht temperatuur (stijgt naarmate product droger wordt)
            outlet_air_temp = inlet_air_temp - 20 * (moisture / initial_moisture) + np.random.normal(0, 0.3)

            # Product temperatuur
            product_temp = outlet_air_temp - 3 + np.random.normal(0, 0.2)

            # Luchtvochtigheid uitlaat
            outlet_humidity = 15 + 30 * (moisture / initial_moisture) + np.random.normal(0, 1)

            # Drukval over bed (daalt als product lichter wordt door drogen)
            pressure_drop = 15 + 5 * (moisture / initial_moisture) + np.random.normal(0, 0.3)  # mbar

            # Luchtdebiet
            airflow = airflow_sp + np.random.normal(0, 10)

            # Energieverbruik
            energy = 0.5 * airflow * (inlet_air_temp - 25) / 1000 + np.random.normal(0, 0.2)  # kW

            # NIR meting (inline, elke 5 minuten = elke 10 punten)
            nir_moisture = moisture + np.random.normal(0, 0.5) if i % 10 == 0 else None

            # Eindpunt detectie
            drying_rate = k_dry * (moisture - target_moisture) if moisture > target_moisture else 0
            phase = "opwarmen" if t_min < 5 else ("constant_rate" if moisture > target_moisture + 5 else ("falling_rate" if moisture > target_moisture + 0.5 else "eindpunt"))

            rows.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_id": f"DRY-{batch:03d}",
                "tijd_sec": t_sec,
                "inlet_lucht_temp_C": round(inlet_air_temp, 1),
                "inlet_lucht_temp_sp_C": inlet_air_temp_sp,
                "outlet_lucht_temp_C": round(outlet_air_temp, 1),
                "product_temp_C": round(product_temp, 1),
                "outlet_luchtvochtigheid_pct": round(outlet_humidity, 1),
                "drukval_mbar": round(pressure_drop, 1),
                "luchtdebiet_m3h": round(airflow, 0),
                "bedmassa_kg": round(bed_mass, 1),
                "vochtgehalte_pct": round(moisture, 2),
                "NIR_vochtgehalte_pct": round(nir_moisture, 2) if nir_moisture is not None else None,
                "droogsnelheid_pctmin": round(drying_rate, 4),
                "energie_kW": round(energy, 2),
                "fase": phase,
            })
            global_idx += 1

    df = pd.DataFrame(rows)
    df.to_csv("data/continue_droging.csv", index=False)
    print(f"continue_droging.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_continuous_mixing():
    """
    Dataset 15: Continue poedermenger - Farmaceutische poedermenging met inline
    NIR-spectroscopie voor uniformiteitsmonitoring. Elke 10 seconden, 48 uur.
    Doel: Voorspel menguniformiteit, detecteer segregatie, procescontrole.
    """
    n = 48 * 3600 // 10  # 48 uur, elke 10 sec
    t = np.arange(n)
    timestamps = [datetime(2025, 3, 1) + timedelta(seconds=int(i * 10)) for i in t]

    # Voedersnelheden (3 componenten)
    api_feed_sp = 5.0  # kg/h (actief ingrediënt)
    excipient_feed_sp = 45.0  # kg/h (vulstof)
    lubricant_feed_sp = 0.5  # kg/h (glijmiddel)

    # Loss-in-weight feeders met variatie en storingen
    api_feed = api_feed_sp + 0.1 * np.sin(2 * np.pi * t / 360) + np.random.normal(0, 0.05, n)
    excipient_feed = excipient_feed_sp + np.random.normal(0, 0.3, n)
    lubricant_feed = lubricant_feed_sp + np.random.normal(0, 0.01, n)

    # Feeder refill events (API feeder, elke ~2 uur)
    refill_interval = 720  # ~2 uur in 10-sec stappen
    for refill_start in range(refill_interval, n, refill_interval):
        duration = np.random.randint(5, 15)
        api_feed[refill_start:refill_start + duration] += np.random.uniform(-0.5, 1.0, min(duration, n - refill_start))

    # Mengersnelheid
    mixer_speed_sp = 250  # RPM
    mixer_speed = mixer_speed_sp + np.random.normal(0, 1, n)

    # Mengertorque (gerelateerd aan dichtheid en vochtigheid)
    torque = 5.0 + 0.01 * mixer_speed + 0.2 * (api_feed / api_feed_sp) + np.random.normal(0, 0.1, n)

    # Temperatuur in menger (wrijving)
    mixer_temp = 22 + 0.02 * mixer_speed + 0.5 * np.cumsum(np.random.normal(0, 0.001, n))
    mixer_temp = np.clip(mixer_temp, 20, 35)

    # Verblijftijd in menger
    residence_time = 120 + np.random.normal(0, 5, n)  # seconden

    # Menguniformiteit (RSD%) - gebaseerd op feeder stabiliteit
    # Lage RSD = goede menging
    api_fraction = api_feed / (api_feed + excipient_feed + lubricant_feed)
    target_fraction = api_feed_sp / (api_feed_sp + excipient_feed_sp + lubricant_feed_sp)

    # Rolling variatie van API fractie -> uniformiteit
    window = 30  # 5 minuten rolling window
    api_rolling_std = pd.Series(api_fraction).rolling(window, min_periods=1).std().values
    rsd = api_rolling_std / target_fraction * 100
    rsd = np.clip(rsd, 0.5, 15)

    # NIR voorspelde uniformiteit (met meetruis)
    nir_rsd = rsd + np.random.normal(0, 0.3, n)
    nir_rsd = np.clip(nir_rsd, 0.3, 20)

    # Injecteer procesverstoringen
    verstoring = np.full(n, "normaal", dtype=object)

    # Verstoring 1: API feeder blokkade (uur 8-8.5)
    block_start = 8 * 360
    block_end = block_start + 180
    api_feed[block_start:block_end] *= 0.3
    rsd[block_start + 10:block_end + 60] = np.clip(rsd[block_start + 10:block_end + 60] * 3, 0, 15)
    nir_rsd[block_start + 10:block_end + 60] = rsd[block_start + 10:block_end + 60] + np.random.normal(0, 0.5, block_end + 60 - block_start - 10)
    verstoring[block_start:block_end + 60] = "feeder_blokkade"

    # Verstoring 2: Segregatie door trillingen (uur 20-22)
    seg_start = 20 * 360
    seg_end = 22 * 360
    rsd[seg_start:seg_end] += 2 * np.sin(np.linspace(0, 4 * np.pi, seg_end - seg_start)) + 1.5
    nir_rsd[seg_start:seg_end] = rsd[seg_start:seg_end] + np.random.normal(0, 0.3, seg_end - seg_start)
    verstoring[seg_start:seg_end] = "segregatie"

    # Verstoring 3: Vochtigheidsprobleem (uur 35-38)
    moist_start = 35 * 360
    moist_end = 38 * 360
    torque[moist_start:moist_end] += 1.5
    mixer_temp[moist_start:moist_end] += 2
    rsd[moist_start:moist_end] += 1.0
    nir_rsd[moist_start:moist_end] = rsd[moist_start:moist_end] + np.random.normal(0, 0.3, moist_end - moist_start)
    verstoring[moist_start:moist_end] = "hoog_vochtgehalte"

    # Kwaliteitsbesluit
    spec_conform = (nir_rsd < 5.0).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "API_voeding_kgh": np.round(api_feed, 3),
        "excipient_voeding_kgh": np.round(excipient_feed, 2),
        "glijmiddel_voeding_kgh": np.round(lubricant_feed, 3),
        "totaal_debiet_kgh": np.round(api_feed + excipient_feed + lubricant_feed, 2),
        "menger_snelheid_RPM": np.round(mixer_speed, 0).astype(int),
        "menger_torque_Nm": np.round(torque, 2),
        "menger_temp_C": np.round(mixer_temp, 1),
        "verblijftijd_sec": np.round(residence_time, 0).astype(int),
        "API_fractie": np.round(api_fraction, 5),
        "RSD_pct": np.round(rsd, 2),
        "NIR_RSD_pct": np.round(nir_rsd, 2),
        "spec_conform": spec_conform,
        "verstoring": verstoring,
    })

    df.to_csv("data/continue_menging.csv", index=False)
    print(f"continue_menging.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_multiunit_process():
    """
    Dataset 16: Geintegreerde productielijn - Chemische plant met reactor, separator,
    en zuiveringssectie. Elke 5 minuten, 60 dagen. Cascade-effecten tussen units.
    Doel: Multivariate procesmonitoring, root cause analysis, cascade fault detection.
    """
    n = 60 * 24 * 12  # 60 dagen, elke 5 min
    t = np.arange(n)
    timestamps = [datetime(2025, 4, 1) + timedelta(minutes=int(i * 5)) for i in t]

    # --- UNIT 1: Reactor ---
    feed_flow = 100 + 5 * np.sin(2 * np.pi * t / (288)) + np.random.normal(0, 1, n)  # kg/h, dagcyclus
    feed_temp = 25 + np.random.normal(0, 0.5, n)  # °C
    feed_conc = 0.95 + np.random.normal(0, 0.01, n)  # mol frac zuiverheid voeding

    reactor_temp = np.zeros(n)
    reactor_temp[0] = 150
    reactor_temp_sp = 150 + 5 * np.sin(2 * np.pi * t / (288 * 7))  # weekcyclus setpoint aanpassing

    for i in range(1, n):
        reactor_temp[i] = reactor_temp[i-1] + 0.03 * (reactor_temp_sp[i] - reactor_temp[i-1]) + np.random.normal(0, 0.1)

    reactor_pressure = 5.0 + 0.01 * (reactor_temp - 150) + np.random.normal(0, 0.05, n)
    reactor_level = 60 + 5 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 1, n)  # %

    conversion = 0.85 + 0.005 * (reactor_temp - 150) - 0.001 * (feed_flow - 100) + np.random.normal(0, 0.01, n)
    conversion = np.clip(conversion, 0.5, 0.99)

    reactor_outlet_flow = feed_flow * (1 + np.random.normal(0, 0.005, n))
    reactor_outlet_conc = feed_conc * (1 - conversion)

    # --- UNIT 2: Flash Separator ---
    sep_temp = reactor_temp - 30 + np.random.normal(0, 0.5, n)
    sep_pressure = 2.0 + np.random.normal(0, 0.03, n)
    sep_level = 50 + 3 * (reactor_outlet_flow - 100) / 5 + np.random.normal(0, 2, n)

    vapor_flow = reactor_outlet_flow * 0.3 * (1 + 0.01 * (sep_temp - 120)) + np.random.normal(0, 0.5, n)
    liquid_flow = reactor_outlet_flow - vapor_flow + np.random.normal(0, 0.3, n)

    # Zuiverheid in vloeibare stroom
    liquid_purity = 0.92 + 0.02 * conversion - 0.005 * (sep_temp - 120) / 10 + np.random.normal(0, 0.005, n)
    liquid_purity = np.clip(liquid_purity, 0.80, 0.99)

    # --- UNIT 3: Zuiveringskolom ---
    column_reflux = 2.5 + np.random.normal(0, 0.05, n)
    column_reboiler_duty = 80 + 5 * (liquid_flow - 70) / 10 + np.random.normal(0, 2, n)
    column_top_temp = 78 + 0.5 * (column_reboiler_duty - 80) / 5 + np.random.normal(0, 0.3, n)
    column_bottom_temp = 120 + 0.3 * (column_reboiler_duty - 80) / 5 + np.random.normal(0, 0.3, n)
    column_dp = 0.3 + 0.01 * liquid_flow / 70 + np.random.normal(0, 0.005, n)

    product_purity = (
        0.995
        + 0.001 * (column_reflux - 2.5) / 0.05
        + 0.0005 * (column_reboiler_duty - 80) / 5
        - 0.002 * (1 - liquid_purity) * 10
        + np.random.normal(0, 0.001, n)
    )
    product_purity = np.clip(product_purity, 0.98, 0.9999)

    product_flow = liquid_flow * liquid_purity + np.random.normal(0, 0.2, n)

    # --- Totaal energieverbruik ---
    total_energy = column_reboiler_duty + 20 + 0.1 * feed_flow + np.random.normal(0, 1, n)

    # --- Procesverstoringen (cascade) ---
    event = np.full(n, "normaal", dtype=object)

    # Event 1: Voedingsverontreiniging (dag 8-9) -> reactor -> separator -> kolom
    ev1_start = 8 * 288
    ev1_end = 9 * 288
    feed_conc[ev1_start:ev1_end] -= 0.05
    conversion[ev1_start:ev1_end] -= 0.03
    liquid_purity[ev1_start + 50:ev1_end + 50] -= 0.02  # vertraagd
    product_purity[ev1_start + 100:ev1_end + 100] -= 0.005  # nog meer vertraagd
    event[ev1_start:ev1_end + 100] = "voedingsverontreiniging"

    # Event 2: Koelwaterprobleem reactor (dag 22-23)
    ev2_start = 22 * 288
    ev2_end = 23 * 288
    reactor_temp[ev2_start:ev2_end] += np.linspace(0, 8, ev2_end - ev2_start)
    reactor_pressure[ev2_start:ev2_end] += 0.3
    conversion[ev2_start:ev2_end] += 0.02
    event[ev2_start:ev2_end] = "koelwater_storing"

    # Event 3: Kolom flooding (dag 40, kort)
    ev3_start = 40 * 288
    ev3_end = 40 * 288 + 72  # 6 uur
    column_dp[ev3_start:ev3_end] *= 2.5
    product_purity[ev3_start:ev3_end] -= 0.008
    column_top_temp[ev3_start:ev3_end] += 5
    event[ev3_start:ev3_end] = "kolom_flooding"

    # Event 4: Geleidelijke katalysator deactivatie (dag 48-56)
    ev4_start = 48 * 288
    ev4_end = 56 * 288
    deact = np.linspace(0, 0.08, ev4_end - ev4_start)
    conversion[ev4_start:ev4_end] -= deact
    event[ev4_start:ev4_end] = "katalysator_deactivatie"

    # Kwaliteitsstatus
    on_spec = (product_purity > 0.990).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        # Unit 1: Reactor
        "R_voeding_flow_kgh": np.round(feed_flow, 1),
        "R_voeding_temp_C": np.round(feed_temp, 1),
        "R_voeding_zuiverheid": np.round(feed_conc, 3),
        "R_temp_sp_C": np.round(reactor_temp_sp, 1),
        "R_temp_actueel_C": np.round(reactor_temp, 2),
        "R_druk_bar": np.round(reactor_pressure, 2),
        "R_niveau_pct": np.round(reactor_level, 1),
        "R_conversie": np.round(conversion, 3),
        # Unit 2: Separator
        "S_temp_C": np.round(sep_temp, 1),
        "S_druk_bar": np.round(sep_pressure, 2),
        "S_niveau_pct": np.round(sep_level, 1),
        "S_damp_flow_kgh": np.round(vapor_flow, 1),
        "S_vloeistof_flow_kgh": np.round(liquid_flow, 1),
        "S_vloeistof_zuiverheid": np.round(liquid_purity, 4),
        # Unit 3: Zuiveringskolom
        "K_reflux_ratio": np.round(column_reflux, 2),
        "K_reboiler_duty_kW": np.round(column_reboiler_duty, 1),
        "K_top_temp_C": np.round(column_top_temp, 1),
        "K_bodem_temp_C": np.round(column_bottom_temp, 1),
        "K_drukval_bar": np.round(column_dp, 4),
        # Product
        "product_zuiverheid": np.round(product_purity, 5),
        "product_flow_kgh": np.round(product_flow, 1),
        "totaal_energie_kW": np.round(total_energy, 1),
        "on_spec": on_spec,
        "event": event,
    })

    df.to_csv("data/geintegreerde_productielijn.csv", index=False)
    print(f"geintegreerde_productielijn.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_compressor_monitoring():
    """
    Dataset 17: Compressormonitoring - Trillingsdata en procesparameters van een
    centrifugaalcompressor. Elke 10 seconden, 90 dagen.
    Doel: Predictive maintenance, trillingsanalyse, degradatie-detectie.
    """
    n_per_day = 8640  # elke 10 sec
    n_days = 90
    n = n_per_day * n_days

    t = np.arange(n)
    timestamps = [datetime(2025, 5, 1) + timedelta(seconds=int(i * 10)) for i in t]

    # Operatiecondities
    speed_sp = 12000  # RPM
    speed = speed_sp + 200 * np.sin(2 * np.pi * t / (n_per_day * 7)) + np.random.normal(0, 20, n)

    suction_pressure = 1.0 + 0.05 * np.sin(2 * np.pi * t / n_per_day) + np.random.normal(0, 0.01, n)
    discharge_pressure = 4.5 + 0.1 * np.sin(2 * np.pi * t / n_per_day) + np.random.normal(0, 0.03, n)
    pressure_ratio = discharge_pressure / suction_pressure

    suction_temp = 25 + 3 * np.sin(2 * np.pi * t / n_per_day) + np.random.normal(0, 0.3, n)
    discharge_temp = suction_temp + 120 + 5 * (pressure_ratio - 4.5) + np.random.normal(0, 1, n)

    flow = 500 + 20 * np.sin(2 * np.pi * t / n_per_day) + np.random.normal(0, 5, n)  # m³/h

    # Lager temperaturen
    bearing_de_temp = 55 + 0.001 * speed / 100 + np.random.normal(0, 0.5, n)  # drive end
    bearing_nde_temp = 52 + 0.001 * speed / 100 + np.random.normal(0, 0.5, n)  # non-drive end

    # Trillingsdata (mm/s RMS)
    # Basislijn trillingen
    vib_de_x = 2.0 + 0.0001 * speed / 100 + np.random.normal(0, 0.1, n)
    vib_de_y = 1.8 + 0.0001 * speed / 100 + np.random.normal(0, 0.1, n)
    vib_nde_x = 1.5 + 0.0001 * speed / 100 + np.random.normal(0, 0.08, n)
    vib_nde_y = 1.4 + 0.0001 * speed / 100 + np.random.normal(0, 0.08, n)

    # Olie parameters
    oil_pressure = 2.5 + np.random.normal(0, 0.05, n)
    oil_temp = 45 + 0.0005 * speed / 100 + np.random.normal(0, 0.3, n)

    # Geleidelijke degradatie van lager (dag 50+)
    degradation_start = 50 * n_per_day
    for i in range(degradation_start, n):
        days_degraded = (i - degradation_start) / n_per_day
        # Exponentiële toename trillingen
        degrad_factor = 1 + 0.02 * days_degraded ** 1.5
        vib_de_x[i] *= degrad_factor
        vib_de_y[i] *= degrad_factor
        bearing_de_temp[i] += 0.1 * days_degraded ** 1.3

    # Olie degradatie (geleidelijk over hele periode)
    oil_quality = 100 - 0.3 * (t / n_per_day) + np.random.normal(0, 1, n)
    oil_quality = np.clip(oil_quality, 30, 100)

    # Surge event (dag 35, kort)
    surge_start = 35 * n_per_day
    surge_end = surge_start + 60  # 10 minuten
    flow[surge_start:surge_end] *= 0.4
    vib_de_x[surge_start:surge_end] *= 5
    vib_de_y[surge_start:surge_end] *= 5
    discharge_pressure[surge_start:surge_end] *= 0.7

    # Status labels
    machine_status = np.full(n, "normaal", dtype=object)
    machine_status[surge_start:surge_end] = "surge"

    for i in range(degradation_start, n):
        days_degraded = (i - degradation_start) / n_per_day
        if days_degraded > 30:
            machine_status[i] = "alarm"
        elif days_degraded > 15:
            machine_status[i] = "waarschuwing"
        elif days_degraded > 0:
            machine_status[i] = "lichte_degradatie"

    # Efficiency
    isentropic_eff = 0.82 - 0.0005 * np.maximum(0, (t - degradation_start) / n_per_day) + np.random.normal(0, 0.005, n)
    isentropic_eff = np.clip(isentropic_eff, 0.5, 0.88)

    power = flow * (discharge_pressure - suction_pressure) / (36 * isentropic_eff) + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "toerental_RPM": np.round(speed, 0).astype(int),
        "aanzuigdruk_bar": np.round(suction_pressure, 3),
        "persdruk_bar": np.round(discharge_pressure, 3),
        "drukverhouding": np.round(pressure_ratio, 2),
        "aanzuigtemp_C": np.round(suction_temp, 1),
        "perstemp_C": np.round(discharge_temp, 1),
        "debiet_m3h": np.round(flow, 1),
        "lager_DE_temp_C": np.round(bearing_de_temp, 1),
        "lager_NDE_temp_C": np.round(bearing_nde_temp, 1),
        "trillingen_DE_x_mms": np.round(vib_de_x, 2),
        "trillingen_DE_y_mms": np.round(vib_de_y, 2),
        "trillingen_NDE_x_mms": np.round(vib_nde_x, 2),
        "trillingen_NDE_y_mms": np.round(vib_nde_y, 2),
        "olie_druk_bar": np.round(oil_pressure, 2),
        "olie_temp_C": np.round(oil_temp, 1),
        "olie_kwaliteit_pct": np.round(oil_quality, 1),
        "isentropisch_rendement": np.round(isentropic_eff, 3),
        "vermogen_kW": np.round(power, 1),
        "machine_status": machine_status,
    })

    # Downsample voor bestandsgrootte (elke minuut i.p.v. 10 sec)
    df = df.iloc[::6].reset_index(drop=True)
    df.to_csv("data/compressor_monitoring.csv", index=False)
    print(f"compressor_monitoring.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_cho_cell_culture():
    """
    Dataset 18: CHO celcultuur perfusie-bioreactor - Productie van monoklonale
    antilichamen (mAb). Continue perfusie over 60 dagen met celretentie.
    Doel: Voorspel titer en celviabiliteit, detecteer cultuurproblemen.
    """
    n_hours = 60 * 24  # 60 dagen, per uur
    t = np.arange(n_hours)
    timestamps = [datetime(2025, 1, 15) + timedelta(hours=int(h)) for h in t]

    # --- Celgroei (logistische groei met doodfase) ---
    viable_cell_density = np.zeros(n_hours)
    total_cell_density = np.zeros(n_hours)
    viability = np.zeros(n_hours)

    vcd = 0.5e6  # cellen/mL start
    tcd = 0.5e6
    mu = 0.03  # groeisnelheid per uur
    k_death = 0.001  # sterfsnelheid per uur
    carrying_capacity = 80e6

    for i in range(n_hours):
        day = i / 24

        # Groeisnelheid neemt af bij hoge dichtheid
        mu_eff = mu * (1 - vcd / carrying_capacity)

        # Sterfsnelheid neemt toe na dag 40
        k_d_eff = k_death * (1 + max(0, (day - 40)) * 0.02)

        # Metabolieten remming
        if i > 0:
            lactate_inhibition = max(0, 1 - lactate[i-1] / 40)
            ammonium_inhibition = max(0, 1 - ammonium[i-1] / 10)
        else:
            lactate_inhibition = 1
            ammonium_inhibition = 1

        mu_eff *= lactate_inhibition * ammonium_inhibition
        mu_eff = max(0, mu_eff)

        vcd = vcd + vcd * (mu_eff - k_d_eff) + np.random.normal(0, 0.05e6)
        vcd = max(0.1e6, vcd)
        tcd = tcd + tcd * mu_eff + np.random.normal(0, 0.03e6)
        tcd = max(vcd, tcd)

        viable_cell_density[i] = vcd
        total_cell_density[i] = tcd
        viability[i] = min(100, max(30, (vcd / tcd) * 100 + np.random.normal(0, 0.5)))

        # Initialiseer metabolieten arrays
        if i == 0:
            glucose = np.zeros(n_hours)
            glutamine = np.zeros(n_hours)
            lactate = np.zeros(n_hours)
            ammonium = np.zeros(n_hours)
            titer = np.zeros(n_hours)

        # Metabolieten
        glucose[i] = max(0, 25 - 0.003 * vcd / 1e6 + np.random.normal(0, 0.3))  # mmol/L
        glutamine[i] = max(0, 4 - 0.0005 * vcd / 1e6 + np.random.normal(0, 0.1))
        lactate[i] = min(40, 0.002 * vcd / 1e6 * (1 + 0.01 * day) + np.random.normal(0, 0.2))
        ammonium[i] = min(10, 0.0003 * vcd / 1e6 * (1 + 0.005 * day) + np.random.normal(0, 0.05))

        # Titer (productie accumuleert)
        specific_productivity = 20e-12 * (1 - 0.005 * max(0, day - 30))  # pg/cel/dag
        titer[i] = (titer[i-1] if i > 0 else 0) + specific_productivity * vcd * 24 / 1e6 / 1000
        titer[i] += np.random.normal(0, 0.01)
        titer[i] = max(0, titer[i])

    # Procesparameters
    temp_sp = np.full(n_hours, 37.0)
    temp_sp[20*24:] = 33.0  # temperatuurshift op dag 20
    temp_actual = temp_sp + np.random.normal(0, 0.05, n_hours)

    ph_sp = np.full(n_hours, 7.0)
    ph_actual = ph_sp + 0.1 * np.sin(2 * np.pi * t / 48) + np.random.normal(0, 0.02, n_hours)

    do_sp = np.full(n_hours, 40.0)  # % luchtverzadiging
    do_actual = do_sp - 5 * viable_cell_density / 50e6 + np.random.normal(0, 1, n_hours)
    do_actual = np.clip(do_actual, 10, 100)

    # Perfusiesnelheid (neemt toe met celdichtheid)
    perfusion_rate = np.clip(0.5 + viable_cell_density / 60e6, 0.5, 3.0) + np.random.normal(0, 0.02, n_hours)  # VVD

    # Bleed rate
    bleed_rate = np.where(viable_cell_density > 50e6, 0.1 + (viable_cell_density - 50e6) / 100e6, 0)
    bleed_rate += np.random.normal(0, 0.005, n_hours)
    bleed_rate = np.clip(bleed_rate, 0, 0.5)

    # Gasflow
    air_sparge = 0.05 + 0.1 * viable_cell_density / 50e6 + np.random.normal(0, 0.005, n_hours)  # L/min
    o2_sparge = np.clip(0.5 * (40 - do_actual) / 10, 0, 1) + np.random.normal(0, 0.01, n_hours)
    co2_sparge = np.clip(0.1 * (ph_actual - 7.0), 0, 0.5) + np.random.normal(0, 0.005, n_hours)

    # Osmolaliteit
    osmolality = 290 + 5 * (lactate / 10) + 3 * (ammonium / 2) + np.random.normal(0, 3, n_hours)

    # Turbiditeit (correleert met celdichtheid)
    turbidity = 0.5 + viable_cell_density / 10e6 + np.random.normal(0, 0.1, n_hours)

    # Celretentie filter druk
    filter_pressure = 0.2 + 0.005 * (t / 24) + np.random.normal(0, 0.01, n_hours)  # bar, stijgt geleidelijk
    filter_pressure = np.clip(filter_pressure, 0.1, 2.0)

    # Events
    event = np.full(n_hours, "normaal", dtype=object)

    # Event 1: pH excursie (dag 12)
    ev_start = 12 * 24
    ph_actual[ev_start:ev_start+6] += np.array([0.1, 0.3, 0.5, 0.4, 0.2, 0.1])
    event[ev_start:ev_start+6] = "pH_excursie"

    # Event 2: DO dip (dag 28, pomp storing)
    ev_start = 28 * 24
    do_actual[ev_start:ev_start+4] = np.array([25, 15, 12, 20])
    event[ev_start:ev_start+4] = "DO_dip"

    # Event 3: Filter fouling (dag 45+)
    ev_start = 45 * 24
    filter_pressure[ev_start:] += 0.3 * np.linspace(0, 1, n_hours - ev_start) ** 2
    event[ev_start:] = np.where(
        filter_pressure[ev_start:] > 1.0, "filter_kritiek",
        np.where(filter_pressure[ev_start:] > 0.7, "filter_fouling", event[ev_start:])
    )

    # Cultuurstatus
    cultuur_fase = np.full(n_hours, "groei", dtype=object)
    for i in range(n_hours):
        day = i / 24
        if day < 2:
            cultuur_fase[i] = "lag"
        elif viability[i] < 70:
            cultuur_fase[i] = "afsterving"
        elif day > 20 and viable_cell_density[i] > 40e6:
            cultuur_fase[i] = "productie"
        elif viable_cell_density[i] > 30e6:
            cultuur_fase[i] = "stationair"

    df = pd.DataFrame({
        "timestamp": timestamps,
        "dag": np.round(t / 24, 2),
        "temp_sp_C": temp_sp,
        "temp_actueel_C": np.round(temp_actual, 2),
        "pH_sp": ph_sp,
        "pH_actueel": np.round(ph_actual, 3),
        "DO_sp_pct": do_sp,
        "DO_actueel_pct": np.round(do_actual, 1),
        "viable_celdichtheid_celmL": np.round(viable_cell_density, 0).astype(int),
        "totale_celdichtheid_celmL": np.round(total_cell_density, 0).astype(int),
        "viabiliteit_pct": np.round(viability, 1),
        "glucose_mmolL": np.round(glucose, 2),
        "glutamine_mmolL": np.round(glutamine, 2),
        "lactaat_mmolL": np.round(lactate, 2),
        "ammonium_mmolL": np.round(ammonium, 3),
        "osmolaliteit_mOsmkg": np.round(osmolality, 0).astype(int),
        "titer_gL": np.round(titer, 3),
        "perfusiesnelheid_VVD": np.round(perfusion_rate, 3),
        "bleed_rate_VVD": np.round(bleed_rate, 4),
        "lucht_sparge_Lmin": np.round(air_sparge, 3),
        "O2_sparge_Lmin": np.round(o2_sparge, 3),
        "CO2_sparge_Lmin": np.round(co2_sparge, 4),
        "turbiditeit_AU": np.round(turbidity, 2),
        "filter_druk_bar": np.round(filter_pressure, 3),
        "cultuur_fase": cultuur_fase,
        "event": event,
    })

    df.to_csv("data/cho_celcultuur_perfusie.csv", index=False)
    print(f"cho_celcultuur_perfusie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_chromatography():
    """
    Dataset 19: Chromatografische zuivering - Downstream processing van mAb.
    Protein A, kationenwisseling en polijststap. Per-run data met UV/conductiviteit traces.
    Doel: Voorspel zuiverheid en opbrengst, optimaliseer elutieprofiel.
    """
    n_runs = 200

    rows = []
    for run in range(n_runs):
        # Kolom en hars eigenschappen
        column_type = np.random.choice(["ProteinA", "CEX", "AEX"])
        column_age_cycles = np.random.randint(1, 300)
        bed_height = np.random.choice([10, 15, 20, 25])  # cm
        column_diameter = np.random.choice([1.0, 1.6, 2.6])  # cm

        # Feed eigenschappen
        feed_titer = np.random.uniform(0.5, 8.0)  # g/L
        feed_volume = np.random.uniform(5, 50)  # CV (column volumes)
        feed_hcp = np.random.uniform(50000, 500000)  # ppm HCP
        feed_dna = np.random.uniform(100, 10000)  # ppm DNA
        feed_aggregate = np.random.uniform(1, 15)  # %

        # Procesparameters
        load_flow_rate = np.random.uniform(100, 400)  # cm/h
        load_ph = {"ProteinA": 7.0, "CEX": 5.0, "AEX": 8.0}[column_type] + np.random.normal(0, 0.1)
        load_conductivity = {"ProteinA": 15, "CEX": 5, "AEX": 8}[column_type] + np.random.normal(0, 0.5)

        wash_volume = np.random.uniform(3, 10)  # CV
        elution_ph = {"ProteinA": 3.5, "CEX": 7.0, "AEX": 5.5}[column_type] + np.random.normal(0, 0.05)
        elution_conductivity = {"ProteinA": 5, "CEX": 25, "AEX": 20}[column_type] + np.random.normal(0, 1)
        elution_flow_rate = np.random.uniform(100, 300)  # cm/h

        # Temperatuur
        temperature = np.random.choice([4, 15, 22]) + np.random.normal(0, 0.5)

        # Resultaten
        # Binding capaciteit daalt met kolom leeftijd
        dbc = 35 * (1 - 0.001 * column_age_cycles) + np.random.normal(0, 2)  # g/L hars

        # Load challenge
        load_challenge = feed_titer * feed_volume / (np.pi * (column_diameter / 2) ** 2 * bed_height / 1000)

        # Doorbraak
        breakthrough = np.clip(load_challenge / dbc * 100, 0, 50)

        # Step yield
        step_yield = (
            90
            - 5 * (load_flow_rate - 200) / 100
            - 0.02 * column_age_cycles
            + 3 * np.log(wash_volume / 5)
            + np.random.normal(0, 3)
        )
        step_yield = np.clip(step_yield, 50, 99)

        # HCP removal (log reduction)
        hcp_lrf = {"ProteinA": 2.5, "CEX": 1.5, "AEX": 1.0}[column_type]
        hcp_lrf += np.random.normal(0, 0.2)
        pool_hcp = feed_hcp / (10 ** hcp_lrf)

        # Aggregate removal
        agg_removal = {"ProteinA": 0.3, "CEX": 0.7, "AEX": 0.5}[column_type]
        pool_aggregate = feed_aggregate * (1 - agg_removal) + np.random.normal(0, 0.3)
        pool_aggregate = np.clip(pool_aggregate, 0.1, 10)

        # UV280 piek (mAU·mL)
        uv_peak_height = step_yield / 100 * feed_titer * 1400 + np.random.normal(0, 50)
        uv_peak_width = 1.5 + 0.5 * (elution_flow_rate - 200) / 100 + np.random.normal(0, 0.1)  # CV
        uv_peak_asymmetry = 1.0 + 0.003 * column_age_cycles + np.random.normal(0, 0.05)

        # Pool volume en concentratie
        pool_volume = uv_peak_width * np.pi * (column_diameter / 2) ** 2 * bed_height / 1000
        pool_concentration = step_yield / 100 * feed_titer * feed_volume / pool_volume if pool_volume > 0 else 0

        # HETP (kolom efficiëntie)
        hetp = 0.03 + 0.0001 * column_age_cycles + np.random.normal(0, 0.003)  # cm

        # Drukval
        delta_p = 0.5 + 0.003 * load_flow_rate + 0.001 * column_age_cycles + np.random.normal(0, 0.05)  # bar

        rows.append({
            "run_id": f"CHROM-{run:04d}",
            "kolom_type": column_type,
            "kolom_leeftijd_cycli": column_age_cycles,
            "bedhoogte_cm": bed_height,
            "kolom_diameter_cm": column_diameter,
            "feed_titer_gL": round(feed_titer, 2),
            "feed_volume_CV": round(feed_volume, 1),
            "feed_HCP_ppm": round(feed_hcp, 0),
            "feed_DNA_ppm": round(feed_dna, 0),
            "feed_aggregaat_pct": round(feed_aggregate, 1),
            "load_snelheid_cmh": round(load_flow_rate, 0),
            "load_pH": round(load_ph, 2),
            "load_conductiviteit_mScm": round(load_conductivity, 1),
            "was_volume_CV": round(wash_volume, 1),
            "elutie_pH": round(elution_ph, 2),
            "elutie_conductiviteit_mScm": round(elution_conductivity, 1),
            "elutie_snelheid_cmh": round(elution_flow_rate, 0),
            "temperatuur_C": round(temperature, 1),
            "DBC_gL": round(dbc, 1),
            "doorbraak_pct": round(breakthrough, 1),
            "drukval_bar": round(delta_p, 2),
            "HETP_cm": round(hetp, 4),
            "UV_piekhoogte_mAU": round(uv_peak_height, 0),
            "UV_piekbreedte_CV": round(uv_peak_width, 2),
            "UV_asymmetrie": round(uv_peak_asymmetry, 2),
            "pool_volume_mL": round(pool_volume, 1),
            "pool_concentratie_gL": round(pool_concentration, 2),
            "pool_HCP_ppm": round(pool_hcp, 0),
            "pool_aggregaat_pct": round(pool_aggregate, 1),
            "stap_opbrengst_pct": round(step_yield, 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/chromatografie_zuivering.csv", index=False)
    print(f"chromatografie_zuivering.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_lyophilization():
    """
    Dataset 20: Lyofilisatie (vriesdroogproces) - Biofarmaceutische formulering.
    Volledige cyclus: invriezen, primair drogen, secundair drogen.
    60 batches met sensordata per minuut (~40 uur per batch).
    Doel: Eindpuntdetectie, procesoptimalisatie, productkwaliteit voorspelling.
    """
    n_batches = 60
    rows = []

    for batch in range(n_batches):
        # Formulering variatie
        solid_content = np.random.uniform(3, 10)  # % w/v
        fill_volume = np.random.choice([0.5, 1.0, 2.0, 5.0])  # mL
        formulation = np.random.choice(["sucrose", "trehalose", "mannitol"])
        protein_conc = np.random.uniform(1, 50)  # mg/mL

        # Cyclus parameters
        freezing_rate = np.random.uniform(0.2, 1.5)  # °C/min
        annealing_temp = np.random.choice([-8, -10, -15, None])
        primary_shelf_temp = np.random.uniform(-25, -10)  # °C
        primary_chamber_pressure = np.random.uniform(50, 200)  # mTorr
        secondary_shelf_temp = np.random.uniform(25, 40)  # °C

        # Collapse temperature (afhankelijk van formulering)
        t_collapse = {"sucrose": -32, "trehalose": -30, "mannitol": -2}[formulation]
        t_collapse += np.random.normal(0, 1)

        # Fases en tijden
        ramp_to_freeze = int(abs(20 - (-40)) / freezing_rate)  # min
        hold_freeze = 120  # min
        anneal_time = 120 if annealing_temp is not None else 0
        primary_dry_time = int(800 + 200 * fill_volume + np.random.normal(0, 60))
        ramp_to_secondary = 120  # min
        secondary_dry_time = int(360 + np.random.normal(0, 30))

        total_time = ramp_to_freeze + hold_freeze + anneal_time + primary_dry_time + ramp_to_secondary + secondary_dry_time

        for i in range(0, total_time, 1):  # per minuut
            phase_time = i

            # Bepaal fase
            if phase_time < ramp_to_freeze:
                phase = "invriezen"
                shelf_temp = 20 - freezing_rate * phase_time
                product_temp = shelf_temp + 2 + np.random.normal(0, 0.3)
                chamber_pressure = 760000  # atmosferisch (mTorr)
                condenser_temp = -60

            elif phase_time < ramp_to_freeze + hold_freeze:
                phase = "hold_bevroren"
                shelf_temp = -40 + np.random.normal(0, 0.2)
                product_temp = -38 + np.random.normal(0, 0.3)
                chamber_pressure = 760000
                condenser_temp = -60

            elif annealing_temp is not None and phase_time < ramp_to_freeze + hold_freeze + anneal_time:
                phase = "annealing"
                progress = (phase_time - ramp_to_freeze - hold_freeze) / anneal_time
                shelf_temp = -40 + (annealing_temp + 40) * min(1, progress * 3) + np.random.normal(0, 0.2)
                product_temp = shelf_temp - 1 + np.random.normal(0, 0.3)
                chamber_pressure = 760000
                condenser_temp = -60

            elif phase_time < ramp_to_freeze + hold_freeze + anneal_time + primary_dry_time:
                phase = "primair_drogen"
                t_in_phase = phase_time - ramp_to_freeze - hold_freeze - anneal_time
                ramp_primary = min(1, t_in_phase / 60)
                shelf_temp = -40 + (-40 - primary_shelf_temp) * (-ramp_primary) + np.random.normal(0, 0.2)
                shelf_temp = max(primary_shelf_temp - 1, min(shelf_temp, -40))
                shelf_temp = primary_shelf_temp + np.random.normal(0, 0.3) if t_in_phase > 60 else shelf_temp

                # Product temperatuur: laag door sublimatie, stijgt naar einde
                sublimation_progress = min(1, t_in_phase / primary_dry_time)
                product_temp = primary_shelf_temp - 15 * (1 - sublimation_progress ** 2) + np.random.normal(0, 0.5)

                chamber_pressure = primary_chamber_pressure + np.random.normal(0, 3)
                condenser_temp = -60 - 10 * (1 - sublimation_progress) + np.random.normal(0, 0.5)

            elif phase_time < ramp_to_freeze + hold_freeze + anneal_time + primary_dry_time + ramp_to_secondary:
                phase = "opwarmen_secundair"
                t_in_phase = phase_time - ramp_to_freeze - hold_freeze - anneal_time - primary_dry_time
                progress = t_in_phase / ramp_to_secondary
                shelf_temp = primary_shelf_temp + (secondary_shelf_temp - primary_shelf_temp) * progress + np.random.normal(0, 0.3)
                product_temp = shelf_temp - 3 * (1 - progress) + np.random.normal(0, 0.3)
                chamber_pressure = primary_chamber_pressure * 0.8 + np.random.normal(0, 2)
                condenser_temp = -65 + np.random.normal(0, 0.5)

            else:
                phase = "secundair_drogen"
                shelf_temp = secondary_shelf_temp + np.random.normal(0, 0.3)
                product_temp = shelf_temp - 1 + np.random.normal(0, 0.2)
                chamber_pressure = primary_chamber_pressure * 0.5 + np.random.normal(0, 2)
                condenser_temp = -65 + np.random.normal(0, 0.5)

            # Pirani vs capacitance manometer ratio (eindpunt indicator)
            if phase in ["primair_drogen", "opwarmen_secundair", "secundair_drogen"]:
                if phase == "primair_drogen":
                    sublimation_progress = min(1, (phase_time - ramp_to_freeze - hold_freeze - anneal_time) / primary_dry_time)
                    pirani_ratio = 1.5 - 0.5 * sublimation_progress ** 3 + np.random.normal(0, 0.02)
                else:
                    pirani_ratio = 1.0 + np.random.normal(0, 0.01)
            else:
                pirani_ratio = None

            rows.append({
                "batch_id": f"LYO-{batch:03d}",
                "tijd_min": i,
                "fase": phase,
                "plaat_temp_C": round(shelf_temp, 1),
                "product_temp_C": round(product_temp, 1),
                "kamer_druk_mTorr": round(max(10, chamber_pressure), 0) if chamber_pressure < 100000 else None,
                "condenser_temp_C": round(condenser_temp, 1),
                "pirani_ratio": round(pirani_ratio, 3) if pirani_ratio is not None else None,
                "formulering": formulation,
                "vaste_stof_pct": round(solid_content, 1),
                "vulvolume_mL": fill_volume,
                "eiwit_conc_mgmL": round(protein_conc, 1),
                "invriessnelheid_Cmin": round(freezing_rate, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/lyofilisatie_cyclus.csv", index=False)
    print(f"lyofilisatie_cyclus.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_pid_control_loops():
    """
    Dataset 21: PID-regelkringen - 20 regelkringen in een chemische plant met
    verschillende tuning-kwaliteit. Elke seconde, 4 uur per loop.
    Doel: Detectie van slecht afgestelde regelaars, oscillatie-detectie, loop performance monitoring.
    """
    n_seconds = 4 * 3600  # 4 uur
    t = np.arange(n_seconds)

    rows = []
    loop_configs = [
        # (naam, type, kwaliteit, setpoint, ruis)
        ("TIC-101", "temperatuur", "goed", 150, 0.3),
        ("TIC-102", "temperatuur", "oscillerend", 80, 0.5),
        ("TIC-103", "temperatuur", "traag", 200, 0.4),
        ("TIC-104", "temperatuur", "agressief", 120, 0.2),
        ("FIC-201", "debiet", "goed", 50, 0.5),
        ("FIC-202", "debiet", "sticking_valve", 30, 0.3),
        ("FIC-203", "debiet", "oscillerend", 75, 0.8),
        ("FIC-204", "debiet", "goed", 100, 1.0),
        ("PIC-301", "druk", "goed", 5, 0.02),
        ("PIC-302", "druk", "traag", 10, 0.05),
        ("PIC-303", "druk", "quantized", 3, 0.01),
        ("LIC-401", "niveau", "goed", 60, 1.0),
        ("LIC-402", "niveau", "oscillerend", 50, 0.8),
        ("LIC-403", "niveau", "saturatie", 70, 1.5),
        ("PH-501", "pH", "goed", 7.0, 0.05),
        ("PH-502", "pH", "niet_lineair", 6.5, 0.08),
        ("AIC-601", "concentratie", "goed", 95, 0.3),
        ("AIC-602", "concentratie", "dood_tijd", 90, 0.5),
        ("TIC-105", "temperatuur", "sensor_ruis", 180, 2.0),
        ("FIC-205", "debiet", "goed", 60, 0.4),
    ]

    for loop_name, loop_type, quality, sp_base, noise in loop_configs:
        # Setpoint (met eventuele veranderingen)
        sp = np.full(n_seconds, float(sp_base))

        # Setpoint veranderingen op specifieke tijdstippen
        sp[3600:] += sp_base * 0.05  # +5% na 1 uur
        sp[7200:] -= sp_base * 0.03  # -3% na 2 uur
        sp[10800:] += sp_base * 0.08  # +8% na 3 uur

        # PV simulatie op basis van kwaliteit
        pv = np.zeros(n_seconds)
        mv = np.zeros(n_seconds)
        error = np.zeros(n_seconds)
        pv[0] = sp[0]
        mv[0] = 50  # %

        for i in range(1, n_seconds):
            error[i] = sp[i] - pv[i-1]

            if quality == "goed":
                # Goede tuning: snel, stabiel
                tau = 30  # tijdconstante
                gain = 1.0
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 100 * error[i], 0, 100)
                pv[i] = pv[i-1] + (sp[i] - pv[i-1]) / tau + np.random.normal(0, noise)

            elif quality == "oscillerend":
                tau = 10
                gain = 4.0  # te hoge gain -> oscillatie
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 50 * error[i], 0, 100)
                pv[i] = pv[i-1] + (mv[i] - mv[i-1]) * sp_base / 200 + np.random.normal(0, noise)
                pv[i] += 0.5 * sp_base * 0.02 * np.sin(2 * np.pi * i / 120)  # 2 min oscillatie

            elif quality == "traag":
                tau = 300  # te hoge tijdconstante
                gain = 0.2
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 500 * error[i], 0, 100)
                pv[i] = pv[i-1] + (sp[i] - pv[i-1]) / tau + np.random.normal(0, noise)

            elif quality == "agressief":
                tau = 5
                gain = 8.0
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]), 0, 100)
                pv[i] = pv[i-1] + (mv[i] - mv[i-1]) * sp_base / 100 + np.random.normal(0, noise * 2)

            elif quality == "sticking_valve":
                tau = 30
                gain = 1.0
                mv_desired = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 100 * error[i], 0, 100)
                # Klep beweegt alleen als signaalverandering > 2%
                if abs(mv_desired - mv[i-1]) > 2.0:
                    mv[i] = mv_desired
                else:
                    mv[i] = mv[i-1]
                pv[i] = pv[i-1] + (mv[i] / 50 - 1) * sp_base * 0.01 + np.random.normal(0, noise)
                pv[i] = pv[i-1] + (sp[i] - pv[i-1]) / (tau * 2) + np.random.normal(0, noise * 1.5)

            elif quality == "quantized":
                tau = 30
                gain = 1.0
                mv_raw = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 100 * error[i], 0, 100)
                mv[i] = round(mv_raw / 0.5) * 0.5  # 0.5% resolutie
                pv[i] = pv[i-1] + (sp[i] - pv[i-1]) / tau + np.random.normal(0, noise)

            elif quality == "saturatie":
                tau = 30
                gain = 1.5
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 80 * error[i], 0, 100)
                # MV raakt gesatureerd
                if i > 5000 and i < 9000:
                    mv[i] = min(mv[i], 98)
                    if mv[i] >= 97:
                        mv[i] = 100  # klep helemaal open
                pv[i] = pv[i-1] + (sp[i] - pv[i-1]) / tau + np.random.normal(0, noise)

            elif quality == "niet_lineair":
                tau = 30
                gain = 1.0
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 100 * error[i], 0, 100)
                # Niet-lineaire klepkarakteristiek
                effective_mv = mv[i] ** 1.5 / 100 ** 0.5
                pv[i] = pv[i-1] + (sp[i] * effective_mv / 50 - pv[i-1]) / tau + np.random.normal(0, noise)
                pv[i] = max(sp_base * 0.5, pv[i])

            elif quality == "dood_tijd":
                tau = 30
                gain = 1.0
                dead_time = 60  # 60 sec dode tijd
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 100 * error[i], 0, 100)
                delayed_mv = mv[max(0, i - dead_time)]
                pv[i] = pv[i-1] + (sp[i] * delayed_mv / 50 - pv[i-1]) / tau + np.random.normal(0, noise)
                pv[i] = max(sp_base * 0.5, pv[i])

            elif quality == "sensor_ruis":
                tau = 30
                gain = 1.0
                mv[i] = np.clip(mv[i-1] + gain * (error[i] - error[i-1]) + gain / 100 * error[i], 0, 100)
                pv[i] = pv[i-1] + (sp[i] - pv[i-1]) / tau + np.random.normal(0, noise)
                # Excessieve sensorruis bovenop
                pv[i] += np.random.normal(0, sp_base * 0.01)

        # Downsample naar elke 5 seconden
        indices = np.arange(0, n_seconds, 5)
        for idx in indices:
            rows.append({
                "loop_tag": loop_name,
                "loop_type": loop_type,
                "tuning_kwaliteit": quality,
                "tijd_sec": int(idx),
                "setpoint": round(float(sp[idx]), 3),
                "proceswaarde": round(float(pv[idx]), 3),
                "regeluitgang_pct": round(float(mv[idx]), 2),
                "afwijking": round(float(sp[idx] - pv[idx]), 4),
                "abs_afwijking": round(float(abs(sp[idx] - pv[idx])), 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/pid_regelkringen.csv", index=False)
    print(f"pid_regelkringen.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_alarm_management():
    """
    Dataset 22: Alarmbeheer - Alarmlogboek van een chemische plant over 6 maanden.
    Doel: Alarm flood detectie, nuisance alarm identificatie, alarm rationalisatie.
    """
    n_days = 180
    alarm_tags = [
        # (tag, prioriteit, type, gemiddeld_per_dag, is_nuisance)
        ("TAH-101", "hoog", "temperatuur", 2, False),
        ("TAL-101", "hoog", "temperatuur", 1, False),
        ("TAH-102", "medium", "temperatuur", 8, True),  # chattering
        ("PAH-201", "hoog", "druk", 3, False),
        ("PAL-201", "medium", "druk", 1, False),
        ("PAH-202", "laag", "druk", 15, True),  # nuisance
        ("FAL-301", "hoog", "debiet", 2, False),
        ("FAH-301", "medium", "debiet", 5, False),
        ("FAL-302", "medium", "debiet", 12, True),  # chattering
        ("LAH-401", "hoog", "niveau", 4, False),
        ("LAL-401", "hoog", "niveau", 3, False),
        ("LAHH-401", "kritiek", "niveau", 0.5, False),
        ("LALL-401", "kritiek", "niveau", 0.3, False),
        ("XA-501", "medium", "equipment", 6, False),
        ("XA-502", "laag", "equipment", 20, True),  # nuisance
        ("AAH-601", "medium", "analyse", 4, False),
        ("AAL-601", "medium", "analyse", 3, False),
        ("SA-701", "hoog", "veiligheid", 0.2, False),
        ("SA-702", "hoog", "veiligheid", 0.1, False),
        ("CA-801", "laag", "communicatie", 25, True),  # nuisance
    ]

    rows = []
    alarm_id = 0

    for day in range(n_days):
        date = datetime(2025, 1, 1) + timedelta(days=day)
        is_weekend = date.weekday() >= 5

        # Alarm flood events (verhoogde alarmfrequentie)
        is_flood = False
        flood_hours = set()
        if np.random.random() < 0.05:  # 5% kans per dag
            flood_start = np.random.randint(0, 20)
            flood_duration = np.random.randint(2, 6)
            flood_hours = set(range(flood_start, flood_start + flood_duration))
            is_flood = True

        for tag, priority, alarm_type, avg_rate, is_nuisance in alarm_tags:
            # Minder alarmen in weekend
            rate = avg_rate * (0.6 if is_weekend else 1.0)

            # Nuisance alarmen chatten meer
            if is_nuisance:
                rate *= 1.5

            n_alarms = np.random.poisson(rate)

            for _ in range(n_alarms):
                hour = np.random.randint(0, 24)
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)

                # Extra alarmen tijdens flood
                in_flood = hour in flood_hours
                if in_flood and np.random.random() < 0.7:
                    n_extra = np.random.randint(1, 5)
                    for extra in range(n_extra):
                        alarm_id += 1
                        rows.append({
                            "alarm_id": alarm_id,
                            "timestamp": (date + timedelta(hours=hour, minutes=minute + extra, seconds=second)).strftime("%Y-%m-%d %H:%M:%S"),
                            "tag": tag,
                            "prioriteit": priority,
                            "type": alarm_type,
                            "status": np.random.choice(["actief", "acknowledged", "teruggekeerd"], p=[0.3, 0.5, 0.2]),
                            "duur_sec": max(1, int(np.random.exponential(300 if is_nuisance else 600))),
                            "is_nuisance": int(is_nuisance),
                            "in_alarm_flood": int(in_flood),
                            "operator_actie": np.random.choice(["acknowledge", "suppress", "actie_genomen", "geen"], p=[0.4, 0.1, 0.3, 0.2]),
                            "dag_type": "weekend" if is_weekend else "werkdag",
                            "shift": "dag" if 6 <= hour < 14 else ("avond" if 14 <= hour < 22 else "nacht"),
                        })

                alarm_id += 1
                rows.append({
                    "alarm_id": alarm_id,
                    "timestamp": (date + timedelta(hours=hour, minutes=minute, seconds=second)).strftime("%Y-%m-%d %H:%M:%S"),
                    "tag": tag,
                    "prioriteit": priority,
                    "type": alarm_type,
                    "status": np.random.choice(["actief", "acknowledged", "teruggekeerd"], p=[0.3, 0.5, 0.2]),
                    "duur_sec": max(1, int(np.random.exponential(300 if is_nuisance else 600))),
                    "is_nuisance": int(is_nuisance),
                    "in_alarm_flood": int(in_flood),
                    "operator_actie": np.random.choice(["acknowledge", "suppress", "actie_genomen", "geen"], p=[0.4, 0.1, 0.3, 0.2]),
                    "dag_type": "weekend" if is_weekend else "werkdag",
                    "shift": "dag" if 6 <= hour < 14 else ("avond" if 14 <= hour < 22 else "nacht"),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv("data/alarmbeheer_logboek.csv", index=False)
    print(f"alarmbeheer_logboek.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_energy_optimization():
    """
    Dataset 23: Energieoptimalisatie - Stoom-, elektriciteits- en koelwaterverbruik
    van een chemische plant met meerdere units. Per 15 minuten, 1 jaar.
    Doel: Energievoorspelling, piekdetectie, optimalisatie van utiliteiten.
    """
    n = 365 * 24 * 4  # 15-min intervallen, 1 jaar
    t = np.arange(n)
    timestamps = [datetime(2025, 1, 1) + timedelta(minutes=int(i * 15)) for i in t]

    hour_of_day = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    is_weekend = (day_of_week >= 5).astype(float)

    # Buitentemperatuur (seizoensgebonden)
    ambient_temp = 10 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 5 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 2, n)

    # Productiebelasting (dag/nacht, weekend)
    production_load = (
        0.7
        + 0.2 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # dagcyclus
        - 0.15 * is_weekend  # weekend lager
        + 0.05 * np.sin(2 * np.pi * day_of_year / 365)  # seizoen
        + np.random.normal(0, 0.03, n)
    )
    production_load = np.clip(production_load, 0.3, 1.0)

    # Stoomverbruik (ton/h)
    steam_demand = (
        30 * production_load
        + 5 * np.maximum(0, 15 - ambient_temp) / 10  # verwarming in winter
        + np.random.normal(0, 1, n)
    )
    steam_demand = np.clip(steam_demand, 5, 50)

    # Stoomdruk
    steam_pressure = 10 + 0.1 * (steam_demand - 25) + np.random.normal(0, 0.1, n)

    # Boiler efficiëntie
    boiler_efficiency = 0.88 - 0.02 * (steam_demand - 25) / 25 + np.random.normal(0, 0.005, n)
    boiler_efficiency = np.clip(boiler_efficiency, 0.78, 0.92)

    # Gasverbruik (Nm³/h)
    gas_consumption = steam_demand * 80 / boiler_efficiency + np.random.normal(0, 10, n)

    # Elektriciteitsverbruik (kW)
    electricity = (
        500 * production_load
        + 100 * np.maximum(0, ambient_temp - 20) / 10  # koeling in zomer
        + 50 * np.maximum(0, 10 - ambient_temp) / 10  # verwarming
        + np.random.normal(0, 15, n)
    )
    electricity = np.clip(electricity, 200, 800)

    # Piekstroom detectie
    peak_threshold = 650
    peak_demand = (electricity > peak_threshold).astype(int)

    # Koelwaterverbruik (m³/h)
    cooling_water = (
        200 * production_load
        + 80 * np.maximum(0, ambient_temp - 15) / 10
        + np.random.normal(0, 10, n)
    )

    # Koeltoren performance
    cooling_tower_approach = 5 + 3 * np.maximum(0, ambient_temp - 20) / 10 + np.random.normal(0, 0.3, n)
    cooling_water_supply_temp = ambient_temp + cooling_tower_approach
    cooling_water_return_temp = cooling_water_supply_temp + 8 * production_load + np.random.normal(0, 0.5, n)

    # Perslucht (Nm³/h)
    compressed_air = 300 * production_load + np.random.normal(0, 10, n)
    compressor_power = compressed_air * 0.15 + np.random.normal(0, 2, n)

    # Stikstof verbruik
    n2_consumption = 50 * production_load + np.random.normal(0, 3, n)

    # Totale energiekost (EUR/h) - vereenvoudigd
    energy_cost = (
        gas_consumption * 0.35  # gas prijs EUR/Nm³
        + electricity * 0.12   # elektra EUR/kWh
        + cooling_water * 0.02  # water EUR/m³
        + peak_demand * 50     # piekboete
    )

    # CO2 uitstoot (kg/h)
    co2_emission = gas_consumption * 2.0 + electricity * 0.4 + np.random.normal(0, 5, n)

    # Onderhoudsstops (4 per jaar, elk 3 dagen)
    maintenance = np.zeros(n, dtype=int)
    for stop_day in [60, 150, 240, 330]:
        start = stop_day * 96
        end = min(start + 3 * 96, n)
        maintenance[start:end] = 1
        production_load[start:end] *= 0.3
        steam_demand[start:end] *= 0.4
        electricity[start:end] *= 0.5

    df = pd.DataFrame({
        "timestamp": timestamps,
        "buitentemp_C": np.round(ambient_temp, 1),
        "productiebelasting": np.round(production_load, 3),
        "stoom_tonh": np.round(steam_demand, 1),
        "stoom_druk_bar": np.round(steam_pressure, 1),
        "boiler_rendement": np.round(boiler_efficiency, 3),
        "gasverbruik_Nm3h": np.round(gas_consumption, 0).astype(int),
        "elektriciteit_kW": np.round(electricity, 0).astype(int),
        "piekbelasting": peak_demand,
        "koelwater_m3h": np.round(cooling_water, 0).astype(int),
        "koelwater_aanvoer_C": np.round(cooling_water_supply_temp, 1),
        "koelwater_retour_C": np.round(cooling_water_return_temp, 1),
        "koeltoren_approach_C": np.round(cooling_tower_approach, 1),
        "perslucht_Nm3h": np.round(compressed_air, 0).astype(int),
        "compressor_vermogen_kW": np.round(compressor_power, 0).astype(int),
        "stikstof_Nm3h": np.round(n2_consumption, 0).astype(int),
        "energiekost_EURh": np.round(energy_cost, 0).astype(int),
        "CO2_kgh": np.round(co2_emission, 0).astype(int),
        "onderhoudsstop": maintenance,
    })

    df.to_csv("data/energie_utiliteiten.csv", index=False)
    print(f"energie_utiliteiten.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_vision_inspection():
    """
    Dataset 24: Machine Vision kwaliteitsinspectie - Geëxtraheerde beeldfeatures
    van visuele inspectie van farmaceutische vials, tabletten en capsules.
    Doel: Defectclassificatie, anomalie-detectie, kwaliteitsvoorspelling.
    """
    n = 5000

    product_type = np.random.choice(["vial", "tablet", "capsule"], n, p=[0.4, 0.35, 0.25])

    # Beeldfeatures (gesimuleerde extractie uit CNN/beeldverwerking)
    # Geometrische features
    area = np.where(product_type == "vial", np.random.normal(15000, 500, n),
           np.where(product_type == "tablet", np.random.normal(8000, 300, n),
                    np.random.normal(6000, 250, n)))
    perimeter = np.sqrt(area) * np.random.uniform(3.4, 3.8, n)
    circularity = 4 * np.pi * area / perimeter ** 2 + np.random.normal(0, 0.02, n)
    aspect_ratio = np.random.normal(1.0, 0.03, n)
    compactness = np.random.normal(0.85, 0.03, n)

    # Kleurfeatures
    mean_intensity = np.random.normal(180, 15, n)
    std_intensity = np.random.normal(12, 3, n)
    color_uniformity = np.random.normal(0.92, 0.03, n)

    # Textuurfeatures (GLCM)
    contrast = np.random.exponential(5, n) + 2
    correlation = np.random.normal(0.85, 0.05, n)
    energy = np.random.normal(0.3, 0.05, n)
    homogeneity = np.random.normal(0.8, 0.04, n)

    # Oppervlaktefeatures
    edge_density = np.random.normal(0.15, 0.03, n)
    roughness = np.random.normal(0.05, 0.01, n)
    n_contours = np.random.poisson(2, n)

    # Defecten genereren (15% defect rate)
    defect_type = np.full(n, "geen", dtype=object)
    label = np.ones(n, dtype=int)  # 1 = goed

    # Kras defect (3%)
    kras_idx = np.random.choice(n, int(n * 0.03), replace=False)
    defect_type[kras_idx] = "kras"
    label[kras_idx] = 0
    edge_density[kras_idx] += np.random.uniform(0.05, 0.15, len(kras_idx))
    contrast[kras_idx] += np.random.uniform(5, 15, len(kras_idx))
    homogeneity[kras_idx] -= np.random.uniform(0.05, 0.1, len(kras_idx))

    # Verkleuring (3%)
    remaining = np.setdiff1d(np.where(label == 1)[0], kras_idx)
    verkl_idx = np.random.choice(remaining, int(n * 0.03), replace=False)
    defect_type[verkl_idx] = "verkleuring"
    label[verkl_idx] = 0
    mean_intensity[verkl_idx] += np.random.uniform(-40, -15, len(verkl_idx))
    color_uniformity[verkl_idx] -= np.random.uniform(0.08, 0.2, len(verkl_idx))
    std_intensity[verkl_idx] += np.random.uniform(5, 15, len(verkl_idx))

    # Barst/breuk (2%)
    remaining = np.where(label == 1)[0]
    barst_idx = np.random.choice(remaining, int(n * 0.02), replace=False)
    defect_type[barst_idx] = "barst"
    label[barst_idx] = 0
    circularity[barst_idx] -= np.random.uniform(0.05, 0.15, len(barst_idx))
    n_contours[barst_idx] += np.random.randint(3, 10, len(barst_idx))
    edge_density[barst_idx] += np.random.uniform(0.1, 0.2, len(barst_idx))
    roughness[barst_idx] += np.random.uniform(0.03, 0.08, len(barst_idx))

    # Deeltje/verontreiniging (3%)
    remaining = np.where(label == 1)[0]
    deeltje_idx = np.random.choice(remaining, int(n * 0.03), replace=False)
    defect_type[deeltje_idx] = "deeltje"
    label[deeltje_idx] = 0
    n_contours[deeltje_idx] += np.random.randint(1, 5, len(deeltje_idx))
    contrast[deeltje_idx] += np.random.uniform(3, 10, len(deeltje_idx))
    std_intensity[deeltje_idx] += np.random.uniform(3, 10, len(deeltje_idx))

    # Vormafwijking (2%)
    remaining = np.where(label == 1)[0]
    vorm_idx = np.random.choice(remaining, int(n * 0.02), replace=False)
    defect_type[vorm_idx] = "vormafwijking"
    label[vorm_idx] = 0
    circularity[vorm_idx] -= np.random.uniform(0.08, 0.2, len(vorm_idx))
    aspect_ratio[vorm_idx] += np.random.uniform(0.05, 0.15, len(vorm_idx))
    compactness[vorm_idx] -= np.random.uniform(0.05, 0.12, len(vorm_idx))

    # Vulniveau probleem (alleen vials, 2%)
    remaining = np.where((label == 1) & (product_type == "vial"))[0]
    vul_idx = np.random.choice(remaining, min(int(n * 0.02), len(remaining)), replace=False)
    defect_type[vul_idx] = "vulniveau"
    label[vul_idx] = 0
    area[vul_idx] *= np.random.uniform(0.85, 0.95, len(vul_idx))

    # Confidence score (hoe zeker is het model)
    confidence = np.where(label == 1,
                          np.random.beta(8, 2, n),
                          np.random.beta(3, 5, n))

    # Lijn metadata
    lijn = np.random.choice(["Lijn_1", "Lijn_2", "Lijn_3"], n)
    snelheid_per_min = np.where(lijn == "Lijn_1", 200, np.where(lijn == "Lijn_2", 300, 150))
    snelheid_per_min = snelheid_per_min + np.random.randint(-10, 10, n)

    df = pd.DataFrame({
        "inspectie_id": [f"VIS-{i:06d}" for i in range(n)],
        "product_type": product_type,
        "productielijn": lijn,
        "lijnsnelheid_per_min": snelheid_per_min,
        "oppervlakte_px": np.round(area, 0).astype(int),
        "omtrek_px": np.round(perimeter, 1),
        "circulariteit": np.round(circularity, 3),
        "aspectverhouding": np.round(aspect_ratio, 3),
        "compactheid": np.round(compactness, 3),
        "gem_intensiteit": np.round(mean_intensity, 1),
        "std_intensiteit": np.round(std_intensity, 1),
        "kleuruniformiteit": np.round(color_uniformity, 3),
        "contrast": np.round(contrast, 2),
        "correlatie": np.round(correlation, 3),
        "energie": np.round(energy, 3),
        "homogeniteit": np.round(homogeneity, 3),
        "randdichtheid": np.round(edge_density, 3),
        "ruwheid": np.round(roughness, 4),
        "aantal_contouren": n_contours,
        "confidence_score": np.round(confidence, 3),
        "defect_type": defect_type,
        "goedgekeurd": label,
    })

    df.to_csv("data/machine_vision_inspectie.csv", index=False)
    print(f"machine_vision_inspectie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_spectroscopy_nir():
    """
    Dataset 25: NIR-spectroscopie multivariate kalibratie - 256 golflengten,
    inline kwaliteitsmetingen voor farmaceutische poeders en tabletten.
    Doel: PLS/PCR kalibratie, wavelength selectie, concentratievoorspelling.
    """
    n = 500
    n_wavelengths = 256
    wavelengths = np.linspace(900, 2500, n_wavelengths)  # nm

    rows = []
    for i in range(n):
        # Echte concentraties
        api_conc = np.random.uniform(5, 50)  # % w/w
        moisture = np.random.uniform(0.5, 8)  # %
        excipient = 100 - api_conc - moisture - np.random.uniform(0, 5)

        # Basislijn NIR spectrum
        baseline = 0.5 + 0.1 * np.sin(2 * np.pi * (wavelengths - 900) / 1600)

        # API absorptiebanden
        api_band1 = api_conc * 0.005 * np.exp(-((wavelengths - 1200) ** 2) / (2 * 30 ** 2))
        api_band2 = api_conc * 0.003 * np.exp(-((wavelengths - 1680) ** 2) / (2 * 40 ** 2))
        api_band3 = api_conc * 0.004 * np.exp(-((wavelengths - 2200) ** 2) / (2 * 35 ** 2))

        # Water absorptiebanden
        water_band1 = moisture * 0.02 * np.exp(-((wavelengths - 1450) ** 2) / (2 * 25 ** 2))
        water_band2 = moisture * 0.015 * np.exp(-((wavelengths - 1940) ** 2) / (2 * 30 ** 2))

        # Excipient band
        exc_band = excipient * 0.001 * np.exp(-((wavelengths - 1550) ** 2) / (2 * 50 ** 2))

        # Deeltjesgrootte effect (scattering)
        particle_size = np.random.uniform(20, 200)  # µm
        scatter = 0.0005 * particle_size * (wavelengths / 1500) ** (-0.5)

        # Totaal spectrum
        spectrum = baseline + api_band1 + api_band2 + api_band3 + water_band1 + water_band2 + exc_band + scatter
        spectrum += np.random.normal(0, 0.005, n_wavelengths)  # instrumentruis

        # Sample metadata
        sample_type = np.random.choice(["poeder", "granulaat", "tablet"])
        batch = f"NIR-B{np.random.randint(1, 30):02d}"

        row = {
            "sample_id": f"NIR-{i:04d}",
            "batch_id": batch,
            "sample_type": sample_type,
            "API_pct": round(api_conc, 2),
            "vochtgehalte_pct": round(moisture, 2),
            "deeltjesgrootte_um": round(particle_size, 0),
        }
        # Spectraaldata
        for j, wl in enumerate(wavelengths):
            row[f"wl_{int(wl)}nm"] = round(float(spectrum[j]), 5)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("data/nir_spectroscopie.csv", index=False)
    print(f"nir_spectroscopie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_valve_diagnostics():
    """
    Dataset 26: Smart Valve diagnostiek - Diagnostische data van regelkleppen
    met verschillende faalmodi. HART/fieldbus parameters.
    Doel: Faalmodusclassificatie, predictive maintenance, klepperformance monitoring.
    """
    n_valves = 50
    n_tests_per_valve = 20  # tests over tijd

    rows = []
    for valve_id in range(n_valves):
        valve_type = np.random.choice(["globe", "butterfly", "ball"])
        size_inch = np.random.choice([2, 4, 6, 8])
        actuator = np.random.choice(["pneumatisch", "elektrisch"])

        # Elke klep krijgt een faalmodus die zich ontwikkelt over tijd
        fault_mode = np.random.choice([
            "gezond", "pakkinglek", "sticking", "erosie",
            "actuator_zwak", "positioner_drift", "cavitatie"
        ], p=[0.35, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1])

        for test in range(n_tests_per_valve):
            age_months = test * 6  # elke 6 maanden een test
            degradation = min(1.0, age_months / 100) if fault_mode != "gezond" else 0

            # Stangafdichting lekkage
            packing_friction = 200 + np.random.normal(0, 20)
            if fault_mode == "pakkinglek":
                packing_friction -= 50 * degradation  # minder wrijving = lek
            elif fault_mode == "sticking":
                packing_friction += 150 * degradation  # meer wrijving

            # Benchmark slag
            travel_deviation = np.random.normal(0, 0.5)
            if fault_mode == "positioner_drift":
                travel_deviation += 3 * degradation

            # Staptijd (sec voor 0-100% slag)
            step_time = {"globe": 5, "butterfly": 3, "ball": 2}[valve_type]
            step_time += np.random.normal(0, 0.3)
            if fault_mode == "actuator_zwak":
                step_time += 5 * degradation
            if fault_mode == "sticking":
                step_time += 3 * degradation

            # Dodeband
            deadband = 1.0 + np.random.normal(0, 0.2)
            if fault_mode == "sticking":
                deadband += 5 * degradation
            if fault_mode == "positioner_drift":
                deadband += 2 * degradation

            # Luchtverbruik (alleen pneumatisch)
            air_consumption = 5 + np.random.normal(0, 0.5) if actuator == "pneumatisch" else 0
            if fault_mode == "pakkinglek" and actuator == "pneumatisch":
                air_consumption += 8 * degradation
            if fault_mode == "actuator_zwak" and actuator == "pneumatisch":
                air_consumption += 3 * degradation

            # Seat leakage
            seat_leakage = 0.01 + np.random.exponential(0.005)
            if fault_mode == "erosie":
                seat_leakage += 0.5 * degradation ** 2

            # Hysterese
            hysteresis = 1.5 + np.random.normal(0, 0.3)
            if fault_mode == "sticking":
                hysteresis += 8 * degradation

            # Signatuur features (gesimuleerde valve signature)
            sig_overshoot = 3 + np.random.normal(0, 1)
            if fault_mode == "actuator_zwak":
                sig_overshoot -= 2 * degradation
            sig_undershoot = 2 + np.random.normal(0, 0.5)
            sig_settling_time = step_time * 1.5 + np.random.normal(0, 0.5)

            # Supply pressure
            supply_pressure = 6.0 + np.random.normal(0, 0.1)
            if fault_mode == "actuator_zwak":
                supply_pressure -= 1.5 * degradation

            # Cycle count
            cycles = age_months * np.random.randint(500, 2000)

            # Temperatuur
            process_temp = np.random.uniform(20, 200)
            if fault_mode == "cavitatie":
                # Cavitatie treedt op bij hoge drukval
                dp_ratio = 0.7 + 0.2 * degradation
            else:
                dp_ratio = np.random.uniform(0.2, 0.6)

            rows.append({
                "klep_id": f"CV-{valve_id:03d}",
                "test_nr": test + 1,
                "leeftijd_maanden": age_months,
                "klep_type": valve_type,
                "diameter_inch": size_inch,
                "actuator_type": actuator,
                "cycli_totaal": cycles,
                "procestemperatuur_C": round(process_temp, 0),
                "pakkingwrijving_N": round(packing_friction, 1),
                "slagafwijking_pct": round(travel_deviation, 2),
                "staptijd_sec": round(step_time, 2),
                "dodeband_pct": round(deadband, 2),
                "hysterese_pct": round(hysteresis, 2),
                "luchtverbruik_Lmin": round(air_consumption, 1),
                "zittinglek_Lmin": round(max(0, seat_leakage), 3),
                "overshoot_pct": round(sig_overshoot, 1),
                "undershoot_pct": round(sig_undershoot, 1),
                "settling_time_sec": round(sig_settling_time, 2),
                "voedingsdruk_bar": round(supply_pressure, 1),
                "dp_ratio": round(dp_ratio, 2),
                "faalmodus": fault_mode,
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/klep_diagnostiek.csv", index=False)
    print(f"klep_diagnostiek.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_cip_cleaning():
    """
    Dataset 27: CIP (Clean-in-Place) validatie - Reinigingsprocessen in farma/food.
    Elke 10 seconden per cyclus, 100 cycli. Conductiviteit, temperatuur, turbiditeit.
    Doel: Eindpuntdetectie, reinigingsefficiëntie voorspelling, procesoptimalisatie.
    """
    n_cycles = 100
    rows = []

    for cycle in range(n_cycles):
        equipment = np.random.choice(["reactor_500L", "reactor_2000L", "tank_5000L", "leiding_DN50"])
        product_prev = np.random.choice(["Product_A", "Product_B", "Product_C", "Product_D"])
        soil_level = np.random.uniform(1, 5)  # relatieve vervuiling

        # CIP parameters (met variatie)
        caustic_conc = np.random.uniform(1.5, 3.0)  # % NaOH
        acid_conc = np.random.uniform(0.5, 1.5)  # % HNO3
        rinse_temp = np.random.uniform(60, 85)  # °C
        flow_rate = {"reactor_500L": 3, "reactor_2000L": 5, "tank_5000L": 8, "leiding_DN50": 1.5}[equipment]
        flow_rate *= np.random.uniform(0.9, 1.1)

        # Fases: voorspoel, loog, tussenspoeling, zuur, naspoeling, eindspoel
        phases = [
            ("voorspoeling", 300, 20, 0, 0),
            ("loogfase", 900, rinse_temp, caustic_conc, 0),
            ("tussenspoeling", 300, rinse_temp * 0.8, 0, 0),
            ("zuurfase", 600, rinse_temp * 0.9, 0, acid_conc),
            ("naspoeling", 300, 40, 0, 0),
            ("eindspoeling", 300, 25, 0, 0),
        ]

        t_global = 0
        for phase_name, duration, temp_target, naoh, hno3 in phases:
            n_points = duration // 10

            for j in range(n_points):
                t_in_phase = j * 10  # seconden

                # Temperatuur
                if j < 6:  # opwarmtijd
                    temp = 20 + (temp_target - 20) * (j / 6)
                else:
                    temp = temp_target + np.random.normal(0, 0.5)

                # Conductiviteit (mS/cm)
                if phase_name == "loogfase":
                    cond = naoh * 50 + np.random.normal(0, 1)
                elif phase_name == "zuurfase":
                    cond = hno3 * 30 + np.random.normal(0, 0.5)
                elif phase_name in ["voorspoeling", "tussenspoeling"]:
                    # Uitspoelen: exponentieel dalend
                    initial_cond = 20 * soil_level if phase_name == "voorspoeling" else naoh * 50
                    cond = initial_cond * np.exp(-0.01 * t_in_phase) + np.random.normal(0, 0.3)
                    cond = max(0.1, cond)
                elif phase_name == "naspoeling":
                    cond = hno3 * 30 * np.exp(-0.015 * t_in_phase) + np.random.normal(0, 0.2)
                    cond = max(0.05, cond)
                else:  # eindspoeling
                    cond = 0.5 + np.random.normal(0, 0.05)  # schoon water

                # Turbiditeit (NTU)
                if phase_name == "voorspoeling":
                    turb = 50 * soil_level * np.exp(-0.008 * t_in_phase) + np.random.normal(0, 1)
                elif phase_name == "loogfase":
                    turb = 10 * soil_level * np.exp(-0.005 * t_in_phase) + np.random.normal(0, 0.5)
                else:
                    turb = 1 + np.random.exponential(0.3)
                turb = max(0.1, turb)

                # pH
                if phase_name == "loogfase":
                    ph = 13 + np.random.normal(0, 0.1)
                elif phase_name == "zuurfase":
                    ph = 2 + np.random.normal(0, 0.1)
                else:
                    ph = 7 + np.random.normal(0, 0.3)

                # TOC (mg/L)
                if phase_name in ["voorspoeling", "loogfase"]:
                    toc = 100 * soil_level * np.exp(-0.005 * (t_global)) + np.random.normal(0, 2)
                else:
                    toc = 2 + np.random.exponential(0.5)
                toc = max(0.1, toc)

                # Flow
                flow = flow_rate + np.random.normal(0, 0.1)

                # Druk
                pressure = 2.0 + 0.5 * flow / flow_rate + np.random.normal(0, 0.05)

                rows.append({
                    "cyclus_id": f"CIP-{cycle:04d}",
                    "equipment": equipment,
                    "vorig_product": product_prev,
                    "vervuilingsgraad": round(soil_level, 1),
                    "fase": phase_name,
                    "tijd_sec": t_global,
                    "tijd_in_fase_sec": t_in_phase,
                    "temperatuur_C": round(temp, 1),
                    "conductiviteit_mScm": round(cond, 2),
                    "turbiditeit_NTU": round(turb, 2),
                    "pH": round(ph, 1),
                    "TOC_mgL": round(toc, 1),
                    "debiet_m3h": round(flow, 2),
                    "druk_bar": round(pressure, 2),
                    "NaOH_pct": round(naoh, 1),
                    "HNO3_pct": round(hno3, 1),
                })
                t_global += 10

        # Eindresultaat van de cyclus: geslaagd als TOC < 5 en conductiviteit < 1
        final_toc = rows[-1]["TOC_mgL"]
        final_cond = rows[-1]["conductiviteit_mScm"]
        result = "geslaagd" if (final_toc < 5 and final_cond < 1.5) else "gefaald"

        # Voeg resultaat toe aan alle rijen van deze cyclus
        for row in rows[-(t_global // 10):]:
            row["cyclus_resultaat"] = result

    df = pd.DataFrame(rows)
    df.to_csv("data/cip_reiniging.csv", index=False)
    print(f"cip_reiniging.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_cleanroom_monitoring():
    """
    Dataset 28: Cleanroom environmental monitoring - Deeltjestelling, temperatuur,
    vochtigheid en druk in farmaceutische cleanrooms. Elke minuut, 90 dagen.
    GMP Grade A/B/C/D classificatie.
    Doel: Classificatie cleanroom status, anomalie-detectie, trend-analyse.
    """
    n_minutes = 90 * 24 * 60  # 90 dagen
    # Downsample naar elke 5 min voor hanteerbare bestandsgrootte
    n = n_minutes // 5
    t = np.arange(n)
    timestamps = [datetime(2025, 6, 1) + timedelta(minutes=int(i * 5)) for i in t]

    rooms = [
        ("Room_A1", "A", 3520, 20, 0.035),
        ("Room_A2", "A", 3520, 20, 0.035),
        ("Room_B1", "B", 3520, 29, 0.025),
        ("Room_B2", "B", 3520, 29, 0.025),
        ("Room_C1", "C", 352000, 2900, 0.015),
        ("Room_D1", "D", 3520000, 29000, 0.010),
    ]

    rows = []
    for room_name, grade, limit_05, limit_50, dp_min in rooms:
        # Basisniveaus (ver onder limiet)
        base_05 = limit_05 * np.random.uniform(0.1, 0.3)
        base_50 = limit_50 * np.random.uniform(0.05, 0.2)

        for i in range(0, n, 6):  # elke 30 min per room (anders te veel data)
            hour = timestamps[i].hour
            day = (timestamps[i] - datetime(2025, 6, 1)).days

            # Dag/nacht effect (meer activiteit overdag)
            activity = 1 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24) if 6 <= hour < 22 else 0.5

            # Personeel effect (Grade A/B gevoeliger)
            personnel_factor = 1 + 0.3 * activity * (0.5 if grade in ["A", "B"] else 0.2)

            # Deeltjestellingen
            particles_05 = base_05 * personnel_factor + np.random.exponential(base_05 * 0.1)
            particles_50 = base_50 * personnel_factor + np.random.exponential(base_50 * 0.05)

            # Temperatuur (strak geregeld)
            temp = 20 + np.random.normal(0, 0.3)

            # Relatieve vochtigheid
            rh = 45 + 5 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 1)

            # Drukverschil (kPa, altijd positief = overdruk)
            dp = dp_min + 0.01 + np.random.normal(0, 0.003)

            # Events die deeltjestelling verhogen
            event = "normaal"

            # Deuropening (random, vaker overdag)
            if np.random.random() < 0.02 * activity:
                particles_05 *= np.random.uniform(2, 5)
                particles_50 *= np.random.uniform(1.5, 3)
                dp -= np.random.uniform(0.005, 0.015)
                event = "deuropening"

            # HVAC storing (zeldzaam, ~1 per maand per room)
            if np.random.random() < 0.0003:
                particles_05 *= np.random.uniform(5, 20)
                particles_50 *= np.random.uniform(3, 10)
                temp += np.random.uniform(1, 3)
                rh += np.random.uniform(3, 8)
                event = "HVAC_storing"

            # Schoonmaak (vroege ochtend)
            if hour == 5 and np.random.random() < 0.3:
                particles_05 *= np.random.uniform(1.5, 3)
                event = "schoonmaak"

            # GMP classificatie check
            in_spec = int(particles_05 <= limit_05 and particles_50 <= limit_50 and dp >= dp_min)

            # Alarmstatus
            if particles_05 > limit_05 or particles_50 > limit_50:
                alarm = "deeltjes_alarm"
            elif dp < dp_min:
                alarm = "druk_alarm"
            elif abs(temp - 20) > 2:
                alarm = "temp_alarm"
            elif rh < 30 or rh > 65:
                alarm = "rv_alarm"
            else:
                alarm = "geen"

            rows.append({
                "timestamp": timestamps[i].strftime("%Y-%m-%d %H:%M"),
                "ruimte": room_name,
                "GMP_grade": grade,
                "deeltjes_05um_per_m3": int(round(particles_05)),
                "deeltjes_50um_per_m3": int(round(particles_50)),
                "limiet_05um": limit_05,
                "limiet_50um": limit_50,
                "temperatuur_C": round(temp, 1),
                "rel_vochtigheid_pct": round(rh, 1),
                "drukverschil_kPa": round(dp, 4),
                "min_drukverschil_kPa": dp_min,
                "in_specificatie": in_spec,
                "alarm": alarm,
                "event": event,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv("data/cleanroom_monitoring.csv", index=False)
    print(f"cleanroom_monitoring.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_golden_batch():
    """
    Dataset 29: Golden Batch analyse - 80 batches van een farmaceutisch coatingproces.
    Vergelijking met ideaal batchprofiel. Tijdreeks per batch.
    Doel: Batch similarity scoring, afwijkingsdetectie, multivariate batch analyse (MPCA).
    """
    n_batches = 80
    n_timepoints = 120  # elke minuut, 2 uur per batch
    t = np.arange(n_timepoints)

    # Golden batch profiel definiëren
    golden_temp = 40 + 15 * (1 - np.exp(-t / 20))  # opwarmprofiel naar 55°C
    golden_spray = np.where(t < 10, t * 0.5, 5.0)  # spray rate opbouw
    golden_pan_speed = np.where(t < 5, 5, np.where(t > 110, 5, 12))  # RPM
    golden_exhaust_temp = golden_temp - 10 - 3 * (golden_spray / 5)
    golden_weight_gain = np.cumsum(golden_spray * 0.01)  # %

    rows = []
    for batch in range(n_batches):
        # Batch kwaliteit (sommige batches wijken af)
        batch_quality = np.random.choice(
            ["goed", "grensgeval", "afwijkend"],
            p=[0.65, 0.20, 0.15]
        )

        # Tablet kern eigenschappen
        core_weight = np.random.normal(350, 5)  # mg
        core_hardness = np.random.normal(80, 8)  # N

        # Coating oplossing
        coating_conc = np.random.normal(15, 0.5)  # % w/w
        coating_viscosity = np.random.normal(50, 5)  # mPa·s

        # Batch-specifieke offsets
        if batch_quality == "goed":
            temp_offset = np.random.normal(0, 0.5)
            spray_offset = np.random.normal(0, 0.2)
            timing_stretch = np.random.uniform(0.95, 1.05)
        elif batch_quality == "grensgeval":
            temp_offset = np.random.normal(0, 2)
            spray_offset = np.random.normal(0, 0.5)
            timing_stretch = np.random.uniform(0.9, 1.1)
        else:  # afwijkend
            temp_offset = np.random.normal(0, 4)
            spray_offset = np.random.normal(0, 1)
            timing_stretch = np.random.uniform(0.8, 1.2)

        for ti in range(n_timepoints):
            t_adj = min(n_timepoints - 1, int(ti * timing_stretch))

            inlet_temp = golden_temp[t_adj] + temp_offset + np.random.normal(0, 0.3)
            spray_rate = max(0, golden_spray[t_adj] + spray_offset + np.random.normal(0, 0.1))
            pan_speed = golden_pan_speed[t_adj] + np.random.normal(0, 0.2)
            exhaust_temp = inlet_temp - 10 - 3 * (spray_rate / 5) + np.random.normal(0, 0.5)
            weight_gain = golden_weight_gain[t_adj] * (1 + spray_offset / 10) + np.random.normal(0, 0.02)

            # Product bed temperatuur
            bed_temp = (inlet_temp + exhaust_temp) / 2 + np.random.normal(0, 0.3)

            # Luchtvochtigheid uitlaat
            exhaust_humidity = 30 + 15 * (spray_rate / 5) + np.random.normal(0, 1)

            # Drukval over trommel
            dp = 3 + 0.5 * (spray_rate / 5) + 0.01 * ti + np.random.normal(0, 0.1)

            # Afwijking van golden batch
            deviation_temp = abs(inlet_temp - golden_temp[min(ti, n_timepoints - 1)])
            deviation_spray = abs(spray_rate - golden_spray[min(ti, n_timepoints - 1)])

            rows.append({
                "batch_id": f"COAT-{batch:03d}",
                "tijd_min": ti,
                "inlet_temp_C": round(inlet_temp, 1),
                "exhaust_temp_C": round(exhaust_temp, 1),
                "bed_temp_C": round(bed_temp, 1),
                "spray_rate_gmin": round(spray_rate, 2),
                "pan_snelheid_RPM": round(pan_speed, 1),
                "exhaust_humidity_pct": round(exhaust_humidity, 1),
                "drukval_mbar": round(dp, 1),
                "gewichtstoename_pct": round(weight_gain, 3),
                "kern_gewicht_mg": round(core_weight, 1),
                "coating_conc_pct": round(coating_conc, 1),
                "afwijking_temp": round(deviation_temp, 2),
                "afwijking_spray": round(deviation_spray, 3),
                "batch_kwaliteit": batch_quality,
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/golden_batch_coating.csv", index=False)
    print(f"golden_batch_coating.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_mpc_process():
    """
    Dataset 30: Model Predictive Control (MPC) - Gegevens van een MPC-geregeld
    destillatieproces met constraints, storingen en setpoint-tracking.
    Doel: MPC performance analyse, constraint analyse, vergelijking met PID.
    """
    n = 48 * 60  # 48 uur, per minuut
    t = np.arange(n)
    timestamps = [datetime(2025, 7, 1) + timedelta(minutes=int(m)) for m in t]

    # Gecontroleerde variabelen (CV's)
    n_cv = 4
    cv_names = ["top_zuiverheid", "bodem_zuiverheid", "kolom_druk", "condenser_niveau"]
    cv_sp = [0.995, 0.005, 1.5, 60]
    cv_lo = [0.990, 0.001, 1.3, 40]
    cv_hi = [1.000, 0.010, 1.7, 80]

    # Gemanipuleerde variabelen (MV's)
    n_mv = 3
    mv_names = ["reflux_flow", "reboiler_duty", "feed_flow"]
    mv_lo = [50, 100, 80]
    mv_hi = [150, 300, 120]

    # Verstorings variabelen (DV's)
    feed_composition = 0.5 + 0.05 * np.sin(2 * np.pi * t / (12 * 60)) + np.random.normal(0, 0.005, n)
    ambient_temp = 25 + 5 * np.sin(2 * np.pi * t / (24 * 60)) + np.random.normal(0, 0.5, n)

    # MPC output simuleren
    cvs = {name: np.zeros(n) for name in cv_names}
    mvs = {name: np.zeros(n) for name in mv_names}

    # Initialisatie
    cvs["top_zuiverheid"][0] = 0.995
    cvs["bodem_zuiverheid"][0] = 0.005
    cvs["kolom_druk"][0] = 1.5
    cvs["condenser_niveau"][0] = 60
    mvs["reflux_flow"][0] = 100
    mvs["reboiler_duty"][0] = 200
    mvs["feed_flow"][0] = 100

    for i in range(1, n):
        # MPC berekent optimale MV aanpassingen
        # Vereenvoudigd: snelle respons op CV afwijkingen met constraints
        for j, (cv_name, sp, lo, hi) in enumerate(zip(cv_names, cv_sp, cv_lo, cv_hi)):
            error = sp - cvs[cv_name][i-1]

            # MV aanpassingen (vereenvoudigd MPC-gedrag)
            if cv_name == "top_zuiverheid":
                mvs["reflux_flow"][i] = np.clip(
                    mvs["reflux_flow"][i-1] + 2 * error * 1000 + np.random.normal(0, 0.3),
                    mv_lo[0], mv_hi[0]
                )
            elif cv_name == "bodem_zuiverheid":
                mvs["reboiler_duty"][i] = np.clip(
                    mvs["reboiler_duty"][i-1] - 5 * error * 1000 + np.random.normal(0, 0.5),
                    mv_lo[1], mv_hi[1]
                )

            if cv_name == "kolom_druk":
                # Druk wordt beinvloed door reboiler en condenser
                pass

        mvs["feed_flow"][i] = np.clip(
            100 + 5 * np.sin(2 * np.pi * i / (8 * 60)) + np.random.normal(0, 0.5),
            mv_lo[2], mv_hi[2]
        )

        # Procesmodel: CV's als functie van MV's en DV's
        cvs["top_zuiverheid"][i] = np.clip(
            cvs["top_zuiverheid"][i-1]
            + 0.00005 * (mvs["reflux_flow"][i] - 100)
            + 0.000002 * (mvs["reboiler_duty"][i] - 200)
            - 0.001 * (feed_composition[i] - 0.5)
            - 0.95 * (cvs["top_zuiverheid"][i-1] - 0.995)
            + np.random.normal(0, 0.0002),
            0.98, 1.0
        )

        cvs["bodem_zuiverheid"][i] = np.clip(
            cvs["bodem_zuiverheid"][i-1]
            - 0.00003 * (mvs["reflux_flow"][i] - 100)
            + 0.00001 * (mvs["reboiler_duty"][i] - 200)
            + 0.0005 * (feed_composition[i] - 0.5)
            - 0.9 * (cvs["bodem_zuiverheid"][i-1] - 0.005)
            + np.random.normal(0, 0.0001),
            0.0, 0.02
        )

        cvs["kolom_druk"][i] = np.clip(
            1.5
            + 0.001 * (mvs["reboiler_duty"][i] - 200)
            - 0.0005 * (ambient_temp[i] - 25)
            + np.random.normal(0, 0.01),
            1.0, 2.0
        )

        cvs["condenser_niveau"][i] = np.clip(
            cvs["condenser_niveau"][i-1]
            + 0.1 * (mvs["reflux_flow"][i-1] - mvs["reflux_flow"][i]) / 5
            - 0.8 * (cvs["condenser_niveau"][i-1] - 60) / 60
            + np.random.normal(0, 0.2),
            20, 95
        )

    # Setpoint veranderingen
    sp_changes = np.zeros(n, dtype=int)
    # SP change op uur 8: top zuiverheid omhoog
    cv_sp_actual = np.full((n, n_cv), cv_sp)
    cv_sp_actual[8*60:, 0] = 0.997
    sp_changes[8*60] = 1
    # SP change op uur 24: feed flow verandering
    sp_changes[24*60] = 1
    # SP change op uur 36: terug naar origineel
    cv_sp_actual[36*60:, 0] = 0.995
    sp_changes[36*60] = 1

    # MPC status
    mpc_status = np.full(n, "actief", dtype=object)
    # MPC uit (operator override) voor 30 min op uur 16
    mpc_status[16*60:16*60+30] = "operator_override"
    # MPC infeasible op uur 30 (constraints te strak)
    mpc_status[30*60:30*60+10] = "infeasible"

    # Constraint violaties
    cv_constraint_violated = np.zeros(n, dtype=int)
    for i in range(n):
        for j, (cv_name, lo, hi) in enumerate(zip(cv_names, cv_lo, cv_hi)):
            if cvs[cv_name][i] < lo or cvs[cv_name][i] > hi:
                cv_constraint_violated[i] = 1

    mv_constraint_violated = np.zeros(n, dtype=int)
    for i in range(n):
        for j, (mv_name, lo, hi) in enumerate(zip(mv_names, mv_lo, mv_hi)):
            if mvs[mv_name][i] <= lo + 0.1 or mvs[mv_name][i] >= hi - 0.1:
                mv_constraint_violated[i] = 1

    # Objectieffunctie waarde (kosten minimalisatie)
    obj_value = mvs["reboiler_duty"] * 0.1 + np.abs(cvs["top_zuiverheid"] - cv_sp_actual[:, 0]) * 10000

    df = pd.DataFrame({
        "timestamp": timestamps,
        # CV's
        "CV_top_zuiverheid": np.round(cvs["top_zuiverheid"], 5),
        "CV_bodem_zuiverheid": np.round(cvs["bodem_zuiverheid"], 5),
        "CV_kolom_druk_bar": np.round(cvs["kolom_druk"], 3),
        "CV_condenser_niveau_pct": np.round(cvs["condenser_niveau"], 1),
        # SP's
        "SP_top_zuiverheid": cv_sp_actual[:, 0],
        "SP_bodem_zuiverheid": cv_sp_actual[:, 1],
        "SP_kolom_druk_bar": cv_sp_actual[:, 2],
        "SP_condenser_niveau_pct": cv_sp_actual[:, 3],
        # MV's
        "MV_reflux_flow_kgh": np.round(mvs["reflux_flow"], 1),
        "MV_reboiler_duty_kW": np.round(mvs["reboiler_duty"], 1),
        "MV_feed_flow_kgh": np.round(mvs["feed_flow"], 1),
        # DV's
        "DV_feed_compositie": np.round(feed_composition, 4),
        "DV_omgevingstemp_C": np.round(ambient_temp, 1),
        # MPC metadata
        "mpc_status": mpc_status,
        "sp_verandering": sp_changes,
        "cv_constraint_overtreding": cv_constraint_violated,
        "mv_constraint_actief": mv_constraint_violated,
        "objectieffunctie": np.round(obj_value, 1),
    })

    df.to_csv("data/mpc_destillatie.csv", index=False)
    print(f"mpc_destillatie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_digital_twin():
    """
    Dataset 31: Digital Twin validatie - Vergelijking van fysiek model (simulatie)
    met werkelijke procesdata van een warmtewisselaarnetwerk.
    Doel: Modelkalibratie, residuanalyse, adaptief modelleren, concept drift detectie.
    """
    n = 30 * 24 * 6  # 30 dagen, elke 10 min
    t = np.arange(n)
    timestamps = [datetime(2025, 8, 1) + timedelta(minutes=int(i * 10)) for i in t]

    # --- Werkelijk proces ---
    feed_flow = 100 + 10 * np.sin(2 * np.pi * t / (144)) + np.random.normal(0, 2, n)
    feed_temp = 25 + 5 * np.sin(2 * np.pi * t / (144 * 7)) + np.random.normal(0, 0.5, n)

    # Werkelijke reactor (niet-lineair, met veroudering)
    aging_factor = 1 - 0.002 * (t / 144)  # geleidelijke degradatie
    aging_factor = np.clip(aging_factor, 0.85, 1.0)

    reactor_temp_real = (
        150
        + 0.3 * (feed_flow - 100)
        + 0.5 * (feed_temp - 25)
        + 10 * np.sin(2 * np.pi * t / 144) * 0.3
        + np.random.normal(0, 0.5, n)
    )

    conversion_real = (
        0.85 * aging_factor
        + 0.002 * (reactor_temp_real - 150)
        - 0.001 * (feed_flow - 100)
        + np.random.normal(0, 0.005, n)
    )
    conversion_real = np.clip(conversion_real, 0.5, 0.99)

    product_temp_real = reactor_temp_real - 30 + np.random.normal(0, 0.3, n)
    energy_real = 50 + 0.3 * feed_flow + 0.1 * reactor_temp_real + np.random.normal(0, 1, n)

    # --- Digital twin model (geidealiseerd, geen aging) ---
    reactor_temp_model = (
        150
        + 0.28 * (feed_flow - 100)  # licht andere coefficient
        + 0.48 * (feed_temp - 25)
        + 10 * np.sin(2 * np.pi * t / 144) * 0.3
    )

    conversion_model = (
        0.85  # geen aging
        + 0.002 * (reactor_temp_model - 150)
        - 0.001 * (feed_flow - 100)
    )
    conversion_model = np.clip(conversion_model, 0.5, 0.99)

    product_temp_model = reactor_temp_model - 30
    energy_model = 50 + 0.3 * feed_flow + 0.1 * reactor_temp_model

    # Residuen (model - werkelijk)
    res_temp = reactor_temp_model - reactor_temp_real
    res_conv = conversion_model - conversion_real
    res_energy = energy_model - energy_real

    # Model confidence (daalt naarmate residuen groeien)
    model_confidence = np.exp(-10 * np.abs(res_conv)) * np.exp(-0.1 * np.abs(res_temp))

    # Concept drift detectie label
    drift_score = np.abs(res_conv) / 0.01 + np.abs(res_temp) / 2
    drift_detected = (drift_score > 3).astype(int)

    # Abrupte procesverandering op dag 20
    ev_start = 20 * 144
    reactor_temp_real[ev_start:] += 3
    conversion_real[ev_start:] -= 0.02
    res_temp[ev_start:] = reactor_temp_model[ev_start:] - reactor_temp_real[ev_start:]
    res_conv[ev_start:] = conversion_model[ev_start:] - conversion_real[ev_start:]

    event = np.full(n, "normaal", dtype=object)
    event[ev_start:] = "katalysator_shift"

    df = pd.DataFrame({
        "timestamp": timestamps,
        "voeding_flow_kgh": np.round(feed_flow, 1),
        "voeding_temp_C": np.round(feed_temp, 1),
        # Werkelijk
        "reactor_temp_werkelijk_C": np.round(reactor_temp_real, 2),
        "conversie_werkelijk": np.round(conversion_real, 4),
        "product_temp_werkelijk_C": np.round(product_temp_real, 1),
        "energie_werkelijk_kW": np.round(energy_real, 1),
        # Model
        "reactor_temp_model_C": np.round(reactor_temp_model, 2),
        "conversie_model": np.round(conversion_model, 4),
        "product_temp_model_C": np.round(product_temp_model, 1),
        "energie_model_kW": np.round(energy_model, 1),
        # Residuen
        "residu_temp_C": np.round(res_temp, 3),
        "residu_conversie": np.round(res_conv, 5),
        "residu_energie_kW": np.round(res_energy, 2),
        # Meta
        "model_confidence": np.round(model_confidence, 3),
        "drift_score": np.round(drift_score, 2),
        "drift_gedetecteerd": drift_detected,
        "event": event,
    })

    df.to_csv("data/digital_twin_validatie.csv", index=False)
    print(f"digital_twin_validatie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_operator_logs():
    """
    Dataset 32: Operator shift logboeken - Gestructureerde en ongestructureerde
    tekst van operatornotities bij shiftoverdracht. NLP-ready.
    Doel: Tekstclassificatie, entity extraction, sentiment/urgentie, trend-detectie.
    """
    n_days = 365
    shifts = ["dag", "avond", "nacht"]

    # Templates voor logboek entries
    normal_templates = [
        "Proces stabiel, alle parameters binnen specificatie. {detail}",
        "Routine operatie, geen bijzonderheden. {detail}",
        "Productie loopt goed, {product} batch {batch} gestart om {time}.",
        "Alle regelkringen in auto. {detail}",
        "Productiewisseling van {product} naar {product2} verlopen zonder problemen.",
        "Preventief onderhoud aan {equipment} uitgevoerd volgens planning.",
        "Monsters genomen voor QC, resultaten binnen spec. {detail}",
        "CIP cyclus {equipment} succesvol afgerond, TOC < 5 ppm.",
    ]

    issue_templates = [
        "STORING: {equipment} uitgevallen om {time}. Oorzaak: {cause}. {action}",
        "AFWIJKING: {parameter} buiten spec ({value}). Actie: {action}",
        "ALARM: {alarm} op {equipment}. Operator heeft {action}",
        "KWALITEIT: Batch {batch} afgekeurd wegens {cause}. Root cause analyse gestart.",
        "LEKKAGE: Kleine lekkage gedetecteerd bij {equipment}. Onderhoud gebeld. {action}",
        "VEILIGHEID: Gasdetectie alarm in zone {zone}. Gebied ontruimd, vals alarm bevestigd.",
        "PROCESSTURING: {parameter} schommelt, PID parameters aangepast. {detail}",
        "GRONDSTOF: Levering {material} vertraagd, productie aangepast. {detail}",
    ]

    equipment_list = ["reactor R-101", "kolom C-201", "pomp P-301", "compressor K-401",
                       "warmtewisselaar E-501", "tank T-601", "filter F-701", "droger D-801",
                       "menger M-901", "centrifuge CF-101"]
    product_list = ["Product_Alpha", "Product_Beta", "Product_Gamma", "Product_Delta"]
    cause_list = ["mechanisch falen", "elektrische storing", "procesverstoring",
                   "hoge temperatuur", "lage druk", "sensor defect", "voedingsprobleem",
                   "verontreiniging in grondstof", "slijtage", "blokkade"]
    action_list = ["handmatig ingegrepen", "naar onderhoud gemeld", "backup systeem ingeschakeld",
                    "productie tijdelijk gestopt", "parameters aangepast", "setpoint verlaagd",
                    "noodprocedure gevolgd", "wacht op onderdelen"]
    parameter_list = ["temperatuur TIC-101", "druk PIC-201", "debiet FIC-301",
                       "niveau LIC-401", "pH AIC-501", "concentratie QIC-601"]
    alarm_list = ["TAH-101 hoge temperatuur", "PAL-201 lage druk", "LAH-401 hoog niveau",
                   "FAL-301 laag debiet", "XA-501 trillingen hoog"]

    rows = []
    log_id = 0

    for day in range(n_days):
        date = datetime(2025, 1, 1) + timedelta(days=day)
        is_weekend = date.weekday() >= 5

        for shift in shifts:
            log_id += 1
            operator = np.random.choice(["Jan V.", "Pieter D.", "Ahmed B.", "Sarah M.",
                                          "Tom K.", "Lisa W.", "Mohammed A.", "Eva S."])

            # Aantal entries per shift
            n_entries = np.random.randint(2, 8)
            entries = []

            # Kans op problemen
            issue_prob = 0.25 if not is_weekend else 0.15

            has_issue = False
            severity = "normaal"
            category = "routine"
            n_alarms = np.random.poisson(3)

            for _ in range(n_entries):
                if np.random.random() < issue_prob and not has_issue:
                    template = np.random.choice(issue_templates)
                    has_issue = True
                    severity = np.random.choice(["laag", "medium", "hoog", "kritiek"],
                                                 p=[0.3, 0.4, 0.2, 0.1])
                    category = np.random.choice(["storing", "kwaliteit", "veiligheid", "proces"])
                    n_alarms += np.random.randint(2, 15)
                else:
                    template = np.random.choice(normal_templates)

                entry = template.format(
                    equipment=np.random.choice(equipment_list),
                    product=np.random.choice(product_list),
                    product2=np.random.choice(product_list),
                    batch=f"B-{np.random.randint(1000, 9999)}",
                    time=f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                    cause=np.random.choice(cause_list),
                    action=np.random.choice(action_list),
                    parameter=np.random.choice(parameter_list),
                    value=f"{np.random.uniform(50, 200):.1f}",
                    alarm=np.random.choice(alarm_list),
                    zone=np.random.randint(1, 6),
                    material=np.random.choice(["NaOH", "HCl", "ethanol", "API grondstof", "verpakkingsmateriaal"]),
                    detail=np.random.choice(["", "Geen verdere actie nodig.",
                                              "Wordt opgevolgd in volgende shift.",
                                              "Logboek bijgewerkt.", "QA geinformeerd."]),
                )
                entries.append(entry.strip())

            full_text = " | ".join(entries)

            # Handover notitie
            handover = np.random.choice([
                "Geen bijzondere aandachtspunten voor volgende shift.",
                "Let op: onderhoud gepland voor morgen.",
                f"Aandachtspunt: {np.random.choice(parameter_list)} moet gemonitord worden.",
                f"Volgende shift: batch {np.random.choice(product_list)} voorbereiden.",
                "Wacht op QC resultaten batch, niet vrijgeven zonder akkoord.",
                f"Storing {np.random.choice(equipment_list)} nog niet opgelost, monitoring vereist.",
            ])

            # Productiecijfers
            production_tons = np.random.uniform(5, 25) * (0.7 if has_issue else 1.0)
            oee = np.random.uniform(0.6, 0.95) * (0.8 if has_issue else 1.0)

            rows.append({
                "log_id": log_id,
                "datum": date.strftime("%Y-%m-%d"),
                "shift": shift,
                "operator": operator,
                "dag_type": "weekend" if is_weekend else "werkdag",
                "logboek_tekst": full_text,
                "handover_notitie": handover,
                "n_entries": n_entries,
                "n_alarmen": n_alarms,
                "productie_ton": round(production_tons, 1),
                "OEE": round(oee, 3),
                "heeft_incident": int(has_issue),
                "ernst": severity,
                "categorie": category,
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/operator_logboeken.csv", index=False)
    print(f"operator_logboeken.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_recipe_optimization():
    """
    Dataset 33: Multi-product receptoptimalisatie - 5 producten op 1 productielijn,
    met wisselende grondstofkwaliteit en seizoenseffecten.
    Doel: Product-specifieke modellering, receptaanpassing, scheduling.
    """
    n_batches = 800

    products = ["Paracetamol_500mg", "Ibuprofen_400mg", "Aspirine_300mg",
                "Metformin_850mg", "Omeprazol_20mg"]
    product_props = {
        "Paracetamol_500mg": {"target_weight": 500, "target_hardness": 80, "coat": True, "complexity": 1.0},
        "Ibuprofen_400mg": {"target_weight": 400, "target_hardness": 70, "coat": True, "complexity": 1.2},
        "Aspirine_300mg": {"target_weight": 300, "target_hardness": 60, "coat": False, "complexity": 0.8},
        "Metformin_850mg": {"target_weight": 850, "target_hardness": 100, "coat": True, "complexity": 1.5},
        "Omeprazol_20mg": {"target_weight": 200, "target_hardness": 50, "coat": True, "complexity": 1.8},
    }

    rows = []
    for batch in range(n_batches):
        product = np.random.choice(products, p=[0.25, 0.25, 0.20, 0.20, 0.10])
        props = product_props[product]

        batch_date = datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365))
        season = ["winter", "lente", "zomer", "herfst"][batch_date.month % 12 // 3]

        # Grondstof variatie (per leverancier, per seizoen)
        supplier = np.random.choice(["Sup_A", "Sup_B", "Sup_C"])
        api_purity = np.random.normal(99.5, 0.3)
        api_moisture = np.random.normal(2.0, 0.3) + (0.5 if season == "zomer" else 0)
        api_particle_d50 = np.random.normal(50, 8)
        api_bulk_density = np.random.normal(0.45, 0.03)

        # Recept parameters (instelbaar)
        granulation_water = np.random.uniform(15, 30)  # % t.o.v. droge massa
        mixing_time = np.random.uniform(5, 20)  # min
        compression_force = np.random.uniform(8, 25)  # kN
        compression_speed = np.random.uniform(20, 60)  # RPM
        coating_time = np.random.uniform(30, 90) if props["coat"] else 0  # min
        drying_temp = np.random.uniform(50, 70)  # °C
        drying_time = np.random.uniform(20, 60)  # min

        # Omgevingscondities
        room_temp = 21 + np.random.normal(0, 1) + (2 if season == "zomer" else (-1 if season == "winter" else 0))
        room_rh = 45 + np.random.normal(0, 3) + (10 if season == "zomer" else (-5 if season == "winter" else 0))

        # Productresultaten (complexe niet-lineaire relaties)
        complexity = props["complexity"]

        weight = (
            props["target_weight"]
            + 2 * (api_moisture - 2) * complexity
            + 0.5 * (granulation_water - 22)
            + np.random.normal(0, 3)
        )

        hardness = (
            props["target_hardness"]
            + 3 * (compression_force - 15)
            - 0.5 * (api_moisture - 2) * 5
            + 0.2 * (api_particle_d50 - 50)
            - 0.3 * (compression_speed - 40) * complexity
            + 1.5 * (mixing_time - 10) / 5
            + np.random.normal(0, 4)
        )

        dissolution = (
            90
            - 1.5 * (compression_force - 15)
            + 0.5 * (api_particle_d50 - 50) / 10
            + 2 * (api_moisture - 2)
            - 0.3 * (hardness - props["target_hardness"])
            + np.random.normal(0, 3)
        )
        dissolution = np.clip(dissolution, 40, 100)

        friability = (
            0.5
            - 0.02 * (compression_force - 15)
            + 0.03 * (api_moisture - 2)
            + 0.01 * (compression_speed - 40)
            + np.random.normal(0, 0.05)
        )
        friability = np.clip(friability, 0.01, 2.0)

        content_uniformity = (
            100
            + 0.5 * (mixing_time - 10) / 5
            - 0.3 * np.abs(api_particle_d50 - 50) / 10
            + np.random.normal(0, 1.5)
        )
        content_uniformity_rsd = np.abs(100 - content_uniformity) + np.random.exponential(0.5)

        moisture_final = max(0.5, api_moisture - 0.03 * drying_temp * drying_time / 100 + np.random.normal(0, 0.2))

        # Coating resultaat
        if props["coat"]:
            weight_gain = 3 + 0.02 * coating_time + np.random.normal(0, 0.2)
            appearance_score = min(5, 4 + 0.01 * coating_time - 0.05 * np.abs(room_rh - 45) + np.random.normal(0, 0.3))
        else:
            weight_gain = 0
            appearance_score = 4 + np.random.normal(0, 0.3)

        # Cyclustijd
        cycle_time = mixing_time + drying_time + 60 + coating_time + np.random.normal(0, 5)

        # Goedkeuring
        approved = int(
            abs(weight - props["target_weight"]) < props["target_weight"] * 0.05
            and hardness > 40 and hardness < 150
            and dissolution > 75
            and friability < 1.0
            and content_uniformity_rsd < 5
            and moisture_final < 3.5
        )

        # Kosten
        batch_cost = (
            100 + cycle_time * 0.5
            + granulation_water * 0.1
            + compression_force * 0.2
            + coating_time * 0.3
            + drying_temp * drying_time * 0.01
        )

        rows.append({
            "batch_id": f"RX-{batch:04d}",
            "datum": batch_date.strftime("%Y-%m-%d"),
            "seizoen": season,
            "product": product,
            "leverancier": supplier,
            "API_zuiverheid_pct": round(api_purity, 2),
            "API_vochtgehalte_pct": round(api_moisture, 2),
            "API_deeltjesgrootte_d50_um": round(api_particle_d50, 1),
            "API_bulkdichtheid_gmL": round(api_bulk_density, 3),
            "granulatie_water_pct": round(granulation_water, 1),
            "mengtijd_min": round(mixing_time, 1),
            "perskracht_kN": round(compression_force, 1),
            "perssnelheid_RPM": round(compression_speed, 0),
            "droogtemp_C": round(drying_temp, 0),
            "droogtijd_min": round(drying_time, 0),
            "coatingtijd_min": round(coating_time, 0),
            "ruimte_temp_C": round(room_temp, 1),
            "ruimte_RV_pct": round(room_rh, 1),
            "gewicht_mg": round(weight, 1),
            "hardheid_N": round(hardness, 1),
            "dissolutie_pct": round(dissolution, 1),
            "brosheid_pct": round(friability, 3),
            "gehalte_uniformiteit_RSD": round(content_uniformity_rsd, 2),
            "restvochtgehalte_pct": round(moisture_final, 2),
            "coating_gewichtstoename_pct": round(weight_gain, 1),
            "uiterlijk_score": round(min(5, max(1, appearance_score)), 1),
            "cyclustijd_min": round(cycle_time, 0),
            "batchkosten_EUR": round(batch_cost, 0),
            "goedgekeurd": approved,
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/recept_optimalisatie.csv", index=False)
    print(f"recept_optimalisatie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_rl_environment():
    """
    Dataset 34: Reinforcement Learning procesomgeving - Reactor temperatuurregeling
    als RL-probleem. States, actions, rewards, next_states voor offline RL.
    Doel: Offline RL, policy evaluation, vergelijking RL vs. PID.
    """
    n_episodes = 200
    n_steps = 300  # stappen per episode (~5 uur, elke minuut)

    rows = []
    for episode in range(n_episodes):
        # Willekeurige initialisatie
        temp = np.random.uniform(140, 160)
        conc = np.random.uniform(0.8, 1.2)  # mol/L
        coolant_temp = np.random.uniform(15, 30)
        setpoint = np.random.choice([145, 150, 155, 160])

        # Controller type (data van verschillende policies)
        policy = np.random.choice(["PID_conservatief", "PID_agressief", "expert", "random"],
                                   p=[0.3, 0.2, 0.3, 0.2])

        cumulative_reward = 0

        for step in range(n_steps):
            # State
            state_temp = temp
            state_conc = conc
            state_coolant = coolant_temp
            state_error = setpoint - temp
            state_energy = 0  # wordt berekend

            # Verstoring
            disturbance = 0.5 * np.sin(2 * np.pi * step / 60) + np.random.normal(0, 0.3)

            # Action: coolant flow aanpassing (-10 tot +10 L/min)
            if policy == "PID_conservatief":
                action = np.clip(0.5 * state_error, -3, 3)
            elif policy == "PID_agressief":
                action = np.clip(2.0 * state_error, -10, 10)
            elif policy == "expert":
                action = np.clip(1.0 * state_error + 0.3 * disturbance, -5, 5)
            else:  # random
                action = np.random.uniform(-10, 10)

            action = round(action, 2)

            # Procesmodel: temperatuur dynamiek
            coolant_flow_base = 50
            coolant_flow = coolant_flow_base + action
            cooling_effect = 0.05 * (temp - coolant_temp) * coolant_flow / coolant_flow_base

            # Exotherme reactie
            reaction_heat = 2.0 * conc * np.exp(-500 / (temp + 273))

            # Temperatuur update
            new_temp = temp + reaction_heat - cooling_effect + disturbance + np.random.normal(0, 0.2)
            new_temp = np.clip(new_temp, 100, 200)

            # Concentratie update
            new_conc = conc - 0.001 * conc * np.exp(-500 / (temp + 273)) + 0.002
            new_conc = np.clip(new_conc, 0.1, 2.0)

            # Coolant temp update
            new_coolant = coolant_temp + 0.01 * (25 - coolant_temp) + np.random.normal(0, 0.1)

            # Energiekosten
            energy = 0.1 * abs(action) + 0.01 * max(0, coolant_flow - 50)

            # Reward
            tracking_error = -abs(new_temp - setpoint) / 5
            energy_penalty = -0.1 * energy
            constraint_penalty = -10 if (new_temp > 180 or new_temp < 120) else 0
            stability_bonus = 0.1 if abs(new_temp - setpoint) < 1 else 0

            reward = tracking_error + energy_penalty + constraint_penalty + stability_bonus
            cumulative_reward += reward

            # Terminal state
            done = int(step == n_steps - 1 or new_temp > 190 or new_temp < 110)

            rows.append({
                "episode": episode,
                "stap": step,
                "policy": policy,
                "setpoint_C": setpoint,
                # State
                "state_temp_C": round(state_temp, 2),
                "state_conc_molL": round(state_conc, 4),
                "state_coolant_C": round(state_coolant, 1),
                "state_error_C": round(state_error, 2),
                "state_verstoring": round(disturbance, 3),
                # Action
                "action_coolant_adj": action,
                # Reward componenten
                "reward_tracking": round(tracking_error, 3),
                "reward_energie": round(energy_penalty, 3),
                "reward_constraint": round(constraint_penalty, 1),
                "reward_totaal": round(reward, 3),
                "cumulatief_reward": round(cumulative_reward, 2),
                # Next state
                "next_temp_C": round(new_temp, 2),
                "next_conc_molL": round(new_conc, 4),
                "next_coolant_C": round(new_coolant, 1),
                "done": done,
            })

            temp, conc, coolant_temp = new_temp, new_conc, new_coolant

            if done and step < n_steps - 1:
                break

    df = pd.DataFrame(rows)
    df.to_csv("data/rl_reactor_control.csv", index=False)
    print(f"rl_reactor_control.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_transfer_learning():
    """
    Dataset 35: Transfer Learning - Dezelfde reactor op 2 locaties (Plant A en B)
    met subtiel verschillende karakteristieken. Plant A heeft veel data, Plant B weinig.
    Doel: Domeinadaptatie, transfer learning, few-shot learning.
    """
    # Plant A: veel data (source domain)
    n_a = 2000
    # Plant B: weinig data (target domain)
    n_b = 100

    rows = []
    for plant, n, offset in [("Plant_A", n_a, 0), ("Plant_B", n_b, 0.1)]:
        for i in range(n):
            temp = np.random.uniform(140, 200)
            pressure = np.random.uniform(1, 10)
            flow = np.random.uniform(50, 150)
            catalyst = np.random.uniform(0.5, 5)
            feed_conc = np.random.uniform(0.5, 2.0)

            # Plant B heeft licht andere karakteristieken
            conversion = (
                0.5 + offset  # baseline verschil
                + 0.002 * (temp - 170) * (1 + offset * 0.5)
                + 0.01 * pressure
                - 0.001 * (flow - 100)
                + 0.05 * np.log1p(catalyst)
                + 0.1 * feed_conc
                - 0.00001 * (temp - 170) ** 2
                + np.random.normal(0, 0.02)
            )
            conversion = np.clip(conversion, 0.1, 0.99)

            # Selectiviteit (ook licht anders)
            selectivity = (
                0.85 - offset * 0.3
                + 0.001 * (temp - 170) * (1 - offset * 0.3)
                - 0.005 * pressure * (1 + offset * 0.2)
                + 0.02 * catalyst
                + np.random.normal(0, 0.015)
            )
            selectivity = np.clip(selectivity, 0.5, 0.99)

            yield_pct = conversion * selectivity * 100

            # Energie (Plant B is ouder, minder efficient)
            energy = (
                30 + 0.2 * flow + 0.5 * temp / 10 + 3 * (1 + offset * 2)
                + np.random.normal(0, 1)
            )

            # Gelabeld (Plant A heeft labels, Plant B deels)
            has_label = True if plant == "Plant_A" else (np.random.random() < 0.3)

            rows.append({
                "sample_id": f"{plant}-{i:05d}",
                "plant": plant,
                "temperatuur_C": round(temp, 1),
                "druk_bar": round(pressure, 2),
                "debiet_kgh": round(flow, 1),
                "katalysator_kgh": round(catalyst, 2),
                "voeding_conc_molL": round(feed_conc, 3),
                "conversie": round(conversion, 4),
                "selectiviteit": round(selectivity, 4),
                "opbrengst_pct": round(yield_pct, 2),
                "energie_kWh": round(energy, 1),
                "heeft_label": int(has_label),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/transfer_learning_reactoren.csv", index=False)
    print(f"transfer_learning_reactoren.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_active_learning():
    """
    Dataset 36: Active Learning - Groot ongelabeld dataset van procesbewaking
    met een klein gelabeld subset. Pool-based active learning scenario.
    Doel: Active learning strategieën, uncertainty sampling, query-by-committee.
    """
    n_total = 5000
    n_labeled = 50  # slechts 1% gelabeld

    # 8 procesfeatures
    temp = np.random.uniform(100, 300, n_total)
    pressure = np.random.uniform(1, 20, n_total)
    flow = np.random.uniform(10, 100, n_total)
    ph = np.random.uniform(4, 10, n_total)
    conductivity = np.random.uniform(0.1, 10, n_total)
    viscosity = np.random.uniform(1, 1000, n_total)
    turbidity = np.random.uniform(0, 100, n_total)
    dissolved_o2 = np.random.uniform(0, 15, n_total)

    # Complexe, niet-lineaire target met meerdere regimes
    regime = np.where(
        (temp > 200) & (pressure > 10), 0,
        np.where((temp < 150) & (ph > 7), 1,
                  np.where(viscosity > 500, 2, 3))
    )

    quality = np.where(
        regime == 0,
        85 + 0.05 * temp - 0.3 * pressure + np.random.normal(0, 3, n_total),
        np.where(
            regime == 1,
            70 + 0.1 * ph * 10 + 0.02 * flow + np.random.normal(0, 4, n_total),
            np.where(
                regime == 2,
                60 + 0.01 * viscosity - 0.5 * turbidity / 10 + np.random.normal(0, 5, n_total),
                75 + 0.03 * temp - 0.1 * pressure + 0.5 * dissolved_o2 + np.random.normal(0, 3, n_total)
            )
        )
    )
    quality = np.clip(quality, 0, 100)

    # Anomalie klasse (5%)
    is_anomaly = np.zeros(n_total, dtype=int)
    anomaly_idx = np.random.choice(n_total, int(n_total * 0.05), replace=False)
    is_anomaly[anomaly_idx] = 1
    quality[anomaly_idx] -= np.random.uniform(15, 30, len(anomaly_idx))

    # Labeling: slechts 50 samples gelabeld (semi-supervised)
    labeled_idx = np.random.choice(n_total, n_labeled, replace=False)
    is_labeled = np.zeros(n_total, dtype=int)
    is_labeled[labeled_idx] = 1

    # Uncertainty features (gesimuleerd van een initieel model)
    # Hoge uncertainty bij grenzen tussen regimes en bij anomalieën
    model_prediction = quality + np.random.normal(0, 5, n_total)
    model_uncertainty = (
        3
        + 5 * np.abs(temp - 175) / 100  # onzekerheid bij regime grenzen
        + 3 * np.abs(pressure - 10) / 10
        + 5 * is_anomaly
        + np.random.exponential(1, n_total)
    )

    # Informativeness score (welke samples zou een active learner kiezen?)
    informativeness = model_uncertainty * (1 + 0.5 * np.abs(quality - 75) / 25)

    df = pd.DataFrame({
        "sample_id": [f"AL-{i:05d}" for i in range(n_total)],
        "temperatuur_C": np.round(temp, 1),
        "druk_bar": np.round(pressure, 2),
        "debiet_Lh": np.round(flow, 1),
        "pH": np.round(ph, 1),
        "conductiviteit_mScm": np.round(conductivity, 2),
        "viscositeit_mPas": np.round(viscosity, 1),
        "turbiditeit_NTU": np.round(turbidity, 1),
        "opgeloste_O2_mgL": np.round(dissolved_o2, 1),
        "kwaliteitsscore": np.round(quality, 1),
        "is_anomalie": is_anomaly,
        "regime": regime,
        "is_gelabeld": is_labeled,
        "model_voorspelling": np.round(model_prediction, 1),
        "model_onzekerheid": np.round(model_uncertainty, 2),
        "informativiteit": np.round(informativeness, 2),
    })

    df.to_csv("data/active_learning_pool.csv", index=False)
    print(f"active_learning_pool.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_nlp_maintenance():
    """
    Dataset 37: NLP Onderhoudslogboek - Gestructureerde en vrije-tekst onderhoudsmeldingen
    voor text mining, classificatie en entity extraction.
    Doel: Tekstclassificatie, NER, prioritering, MTBF/MTTR analyse.
    """
    n = 3000

    equipment_systems = {
        "mechanisch": ["pomp", "compressor", "roerder", "transportband", "centrifuge", "ventilator"],
        "elektrisch": ["motor", "frequentieregelaar", "transformator", "schakelaar", "sensor", "PLC"],
        "instrumentatie": ["flowmeter", "druktransmitter", "temperatuursensor", "niveaumeter", "analyzer", "klep"],
        "piping": ["leiding", "flens", "pakking", "expansievat", "filter", "appendage"],
    }

    failure_modes = {
        "pomp": ["cavitatie", "afdichtingslek", "lager defect", "impeller slijtage", "motor oververhitting"],
        "compressor": ["surge", "lager defect", "klep defect", "olie lek", "trillingen hoog"],
        "motor": ["oververhitting", "isolatie defect", "lager defect", "onbalans", "overbelasting"],
        "klep": ["sticking", "lekkage", "actuator defect", "positioner drift", "erosie"],
        "sensor": ["drift", "bevriezing", "kortsluiting", "kalibratie fout", "breuk"],
    }

    action_descriptions = [
        "Vervangen door nieuw onderdeel, getest en vrijgegeven.",
        "Gerepareerd ter plaatse, lager vervangen en uitgelijnd.",
        "Tijdelijke reparatie uitgevoerd, definitieve reparatie gepland voor volgende stop.",
        "Gekalibreerd volgens SOP-{sop}, resultaten binnen spec.",
        "Schoongemaakt en geinspecteerd, geen verdere actie nodig.",
        "Onderdeel besteld, verwachte levertijd {days} dagen. Bypass ingeschakeld.",
        "Root cause analyse uitgevoerd: {cause}. Preventieve maatregel ingevoerd.",
        "Storingsanalyse met trilingsmetingen uitgevoerd. Aanbeveling: vervangen bij volgende stop.",
        "Elektrische metingen uitgevoerd: isolatie {value} MOhm, binnen spec.",
        "Lekkage afgedicht met nieuwe pakking. Leidingtest uitgevoerd bij {pressure} bar.",
    ]

    rows = []
    for i in range(n):
        date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 730))
        system = np.random.choice(list(equipment_systems.keys()), p=[0.35, 0.25, 0.25, 0.15])
        equipment_type = np.random.choice(equipment_systems[system])
        tag = f"{equipment_type[:3].upper()}-{np.random.randint(100, 999)}"
        location = np.random.choice(["Unit_100", "Unit_200", "Unit_300", "Unit_400", "Utiliteiten"])

        # Failure mode
        if equipment_type in failure_modes:
            failure = np.random.choice(failure_modes[equipment_type])
        else:
            failure = np.random.choice(["slijtage", "lekkage", "defect", "storing", "veroudering"])

        # Werkorder type
        wo_type = np.random.choice(
            ["correctief", "preventief", "predictief", "verbetering", "inspectie"],
            p=[0.35, 0.30, 0.15, 0.10, 0.10]
        )

        # Prioriteit
        priority = np.random.choice(["laag", "medium", "hoog", "kritiek"],
                                     p=[0.2, 0.4, 0.3, 0.1])

        # Tijden
        response_time_h = np.random.exponential({"kritiek": 1, "hoog": 4, "medium": 12, "laag": 48}[priority])
        repair_time_h = np.random.exponential({"correctief": 6, "preventief": 4, "predictief": 3, "verbetering": 8, "inspectie": 2}[wo_type])
        downtime_h = repair_time_h * np.random.uniform(0.5, 1.5) if wo_type == "correctief" else repair_time_h * 0.3

        # Kosten
        material_cost = np.random.exponential(200) + 50
        labor_cost = repair_time_h * np.random.uniform(50, 80)

        # Vrije tekst beschrijving
        description_templates = [
            f"Storing op {tag} ({equipment_type}) in {location}. Symptoom: {failure}. ",
            f"Melding ontvangen: {equipment_type} {tag} functioneert niet correct. ",
            f"Tijdens ronde geconstateerd: {failure} bij {tag}. ",
            f"Operator meldt abnormaal geluid/trilling bij {tag} ({equipment_type}). ",
            f"Alarming op DCS voor {tag}: {failure} gedetecteerd. ",
        ]
        description = np.random.choice(description_templates)

        action = np.random.choice(action_descriptions).format(
            sop=np.random.randint(100, 999),
            days=np.random.randint(1, 30),
            cause=failure,
            value=np.random.uniform(50, 500),
            pressure=np.random.uniform(5, 40),
        )

        full_text = description + action

        # Herhaling (recurring failure)
        is_recurring = int(np.random.random() < 0.15)

        rows.append({
            "werkorder_id": f"WO-{i:05d}",
            "datum": date.strftime("%Y-%m-%d"),
            "tag": tag,
            "equipment_type": equipment_type,
            "systeem": system,
            "locatie": location,
            "werkorder_type": wo_type,
            "prioriteit": priority,
            "faalmodus": failure,
            "beschrijving": full_text,
            "responstijd_uur": round(response_time_h, 1),
            "reparatietijd_uur": round(repair_time_h, 1),
            "stilstandtijd_uur": round(downtime_h, 1),
            "materiaalkosten_EUR": round(material_cost, 0),
            "arbeidskosten_EUR": round(labor_cost, 0),
            "totaalkosten_EUR": round(material_cost + labor_cost, 0),
            "terugkerend": is_recurring,
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/nlp_onderhoudslogboek.csv", index=False)
    print(f"nlp_onderhoudslogboek.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_virtual_metrology():
    """
    Dataset 38: Virtual Metrology - Voorspelling van productkwaliteit uit procesdata
    zonder offline lab-meting. Semiconductor/farma-inspired.
    Doel: Virtuele meting, semi-supervised learning, missing data handling.
    """
    n = 3000

    # Procesdata (altijd beschikbaar)
    temp_zone1 = np.random.normal(200, 5, n)
    temp_zone2 = np.random.normal(210, 5, n)
    temp_zone3 = np.random.normal(220, 5, n)
    pressure = np.random.normal(5, 0.3, n)
    gas_flow_1 = np.random.normal(100, 5, n)
    gas_flow_2 = np.random.normal(50, 3, n)
    power = np.random.normal(500, 20, n)
    time_in_process = np.random.uniform(30, 120, n)
    chamber_humidity = np.random.normal(2, 0.5, n)
    substrate_temp = np.random.normal(180, 3, n)

    # Kwaliteitsmetingen (duur, traag - maar ground truth)
    # Dikte (nm) - primaire target
    thickness = (
        100
        + 0.5 * (temp_zone2 - 210)
        + 2 * (time_in_process - 75) / 45
        + 0.3 * (gas_flow_1 - 100) / 5
        - 0.1 * (pressure - 5) / 0.3
        + 0.05 * (power - 500) / 20
        + np.random.normal(0, 1.5, n)
    )

    # Uniformiteit (%) - secundaire target
    uniformity = (
        95
        - 0.3 * np.abs(temp_zone1 - temp_zone3) / 5
        + 0.1 * (gas_flow_2 - 50) / 3
        - 0.2 * (chamber_humidity - 2) / 0.5
        + np.random.normal(0, 1, n)
    )
    uniformity = np.clip(uniformity, 80, 100)

    # Stress (MPa) - tertiaire target
    stress = (
        50
        + 2 * (temp_zone2 - 210)
        - 1.5 * (substrate_temp - 180)
        + 0.5 * (pressure - 5) / 0.3
        + np.random.normal(0, 5, n)
    )

    # Lab metingen zijn slechts 10% beschikbaar (duur/tijdrovend)
    lab_measured = np.zeros(n, dtype=int)
    lab_idx = np.sort(np.random.choice(n, int(n * 0.1), replace=False))
    lab_measured[lab_idx] = 1

    # In-line metingen (altijd beschikbaar, maar minder nauwkeurig)
    inline_thickness = thickness + np.random.normal(0, 5, n)  # grotere ruis
    inline_reflectance = 0.3 + 0.002 * thickness + np.random.normal(0, 0.01, n)

    # Kamer-specifieke effecten (concept drift)
    chamber = np.random.choice(["Chamber_A", "Chamber_B", "Chamber_C"], n)
    chamber_offset = {"Chamber_A": 0, "Chamber_B": 2, "Chamber_C": -1.5}
    thickness += np.array([chamber_offset[c] for c in chamber])

    # Onderhoudscycli (na onderhoud wijzigt proces licht)
    maintenance_cycle = np.random.randint(1, 50, n)
    thickness += 0.02 * maintenance_cycle  # geleidelijke drift

    # Wafer/batch positie effect
    position = np.random.choice(["center", "edge", "corner"], n, p=[0.4, 0.4, 0.2])
    position_offset = {"center": 0, "edge": -1, "corner": -2}
    uniformity += np.array([position_offset[p] for p in position])

    df = pd.DataFrame({
        "sample_id": [f"VM-{i:05d}" for i in range(n)],
        "kamer": chamber,
        "positie": position,
        "onderhoudscyclus": maintenance_cycle,
        "temp_zone1_C": np.round(temp_zone1, 1),
        "temp_zone2_C": np.round(temp_zone2, 1),
        "temp_zone3_C": np.round(temp_zone3, 1),
        "druk_mbar": np.round(pressure, 2),
        "gasflow_1_sccm": np.round(gas_flow_1, 1),
        "gasflow_2_sccm": np.round(gas_flow_2, 1),
        "vermogen_W": np.round(power, 0).astype(int),
        "procestijd_min": np.round(time_in_process, 1),
        "kamer_vochtigheid_pct": np.round(chamber_humidity, 2),
        "substraat_temp_C": np.round(substrate_temp, 1),
        "inline_dikte_nm": np.round(inline_thickness, 1),
        "inline_reflectantie": np.round(inline_reflectance, 4),
        "lab_dikte_nm": np.where(lab_measured, np.round(thickness, 2), np.nan),
        "lab_uniformiteit_pct": np.where(lab_measured, np.round(uniformity, 1), np.nan),
        "lab_stress_MPa": np.where(lab_measured, np.round(stress, 1), np.nan),
        "lab_gemeten": lab_measured,
    })

    df.to_csv("data/virtual_metrology.csv", index=False)
    print(f"virtual_metrology.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_extruder():
    """
    Dataset 39: Hot-melt extrusie - Farmaceutische continue extrusie.
    Twin-screw extruder met meerdere barrel zones. Elke 5 seconden, 24 uur.
    Doel: Procesmonitoring, soft sensor, residence time modellering.
    """
    n = 24 * 3600 // 5  # 24 uur, elke 5 sec
    t = np.arange(n)
    timestamps = [datetime(2025, 9, 1) + timedelta(seconds=int(i * 5)) for i in t]

    # Schroef configuratie
    screw_speed_sp = 200  # RPM
    screw_speed = screw_speed_sp + 20 * np.sin(2 * np.pi * t / (720)) + np.random.normal(0, 1, n)

    # Voeding
    feed_rate_sp = 5.0  # kg/h
    feed_rate = feed_rate_sp + 0.3 * np.sin(2 * np.pi * t / 360) + np.random.normal(0, 0.05, n)

    # 8 barrel zones temperatuur (setpoint + actueel)
    n_zones = 8
    zone_sp = [40, 80, 120, 150, 160, 170, 165, 155]
    zone_temps = {}
    for z in range(n_zones):
        sp = zone_sp[z]
        actual = sp + np.random.normal(0, 0.5, n)
        # Thermische interactie tussen zones
        if z > 0:
            actual += 0.05 * (zone_temps[f"zone{z}_temp_C"] - sp)
        zone_temps[f"zone{z+1}_temp_C"] = actual
        zone_temps[f"zone{z+1}_temp_sp_C"] = np.full(n, sp)

    # Schroef torque (% van max)
    torque = (
        30
        + 0.1 * screw_speed / 10
        + 5 * feed_rate / feed_rate_sp
        + 2 * np.sin(2 * np.pi * t / 720)
        + np.random.normal(0, 0.5, n)
    )
    torque = np.clip(torque, 10, 80)

    # Specific mechanical energy (kWh/kg)
    sme = 2 * np.pi * screw_speed * torque / (60 * 100 * feed_rate) * 0.5
    sme += np.random.normal(0, 0.01, n)

    # Die druk (bar)
    die_pressure = (
        30
        + 5 * feed_rate / feed_rate_sp
        - 0.05 * screw_speed
        + 2 * np.sin(2 * np.pi * t / 360)
        + np.random.normal(0, 0.5, n)
    )
    die_pressure = np.clip(die_pressure, 5, 80)

    # Smelttemperatuur aan die
    melt_temp = zone_sp[-1] + 5 + 0.02 * screw_speed + 0.5 * sme * 10 + np.random.normal(0, 0.5, n)

    # Product kwaliteit
    # API content (bepaald door mixing)
    api_content = 30 + 0.01 * sme * 100 - 0.005 * np.abs(screw_speed - 200) + np.random.normal(0, 0.3, n)

    # Degradatie (hoger bij hoge temp en lang verblijf)
    degradation = 0.1 + 0.005 * (melt_temp - 160) + 0.001 * torque + np.random.exponential(0.05, n)
    degradation = np.clip(degradation, 0, 5)

    # Inline Raman (elke 30 sec = elke 6 punten)
    raman_api = api_content + np.random.normal(0, 0.5, n)
    raman_measured = np.zeros(n, dtype=int)
    raman_measured[::6] = 1

    # Procesverstoringen
    event = np.full(n, "normaal", dtype=object)

    # Feeder puls
    ev1 = 5000
    feed_rate[ev1:ev1+30] *= 1.5
    die_pressure[ev1:ev1+50] += 10
    event[ev1:ev1+50] = "feeder_puls"

    # Barrel zone heater fout
    ev2 = 10000
    zone_temps["zone5_temp_C"][ev2:ev2+200] -= np.linspace(0, 15, 200)
    event[ev2:ev2+200] = "heater_fout"

    # Materiaal verandering
    ev3 = 14000
    torque[ev3:ev3+500] += 5
    sme[ev3:ev3+500] += 0.03
    event[ev3:ev3+500] = "materiaal_batch_verschil"

    data = {
        "timestamp": timestamps,
        "schroefsnelheid_RPM": np.round(screw_speed, 0).astype(int),
        "voeding_kgh": np.round(feed_rate, 3),
        "torque_pct": np.round(torque, 1),
        "SME_kWhkg": np.round(sme, 4),
        "die_druk_bar": np.round(die_pressure, 1),
        "smelt_temp_C": np.round(melt_temp, 1),
    }

    for z in range(n_zones):
        data[f"zone{z+1}_temp_C"] = np.round(zone_temps[f"zone{z+1}_temp_C"], 1)
        data[f"zone{z+1}_sp_C"] = zone_sp[z]

    data.update({
        "API_gehalte_pct": np.round(api_content, 2),
        "degradatie_pct": np.round(degradation, 3),
        "Raman_API_pct": np.where(raman_measured, np.round(raman_api, 2), np.nan),
        "Raman_gemeten": raman_measured,
        "event": event,
    })

    df = pd.DataFrame(data)
    df.to_csv("data/extrusie_hotmelt.csv", index=False)
    print(f"extrusie_hotmelt.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_membrane_filtration():
    """
    Dataset 40: Membraanfiltratie monitoring - UF/RO waterzuivering.
    Transmembraandruk, flux, fouling over 6 maanden. Reinigingscycli.
    Doel: Fouling voorspelling, reinigingsmoment optimalisatie, flux modellering.
    """
    n_hours = 180 * 24  # 6 maanden, per uur
    t = np.arange(n_hours)
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=int(h)) for h in t]

    # 4 membraanmodules
    rows = []
    for module_id in range(4):
        module_type = ["RO", "UF", "RO", "NF"][module_id]
        membrane_age_days = np.random.randint(0, 365)

        # Voedingswater kwaliteit (seizoensgebonden)
        day_of_year = np.array([(datetime(2025, 1, 1) + timedelta(hours=int(h))).timetuple().tm_yday for h in t])
        feed_tds = 500 + 100 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 20, n_hours)
        feed_turbidity = 2 + 1.5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.exponential(0.3, n_hours)
        feed_temp = 15 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 1, n_hours)
        feed_ph = 7.2 + 0.3 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 0.1, n_hours)
        feed_pressure = {"RO": 15, "UF": 2, "NF": 8}[module_type] + np.random.normal(0, 0.2, n_hours)

        # Fouling opbouw (niet-lineair, met reinigingscycli)
        fouling_resistance = np.zeros(n_hours)
        last_cleaning = 0
        cleaning_events = []

        for i in range(n_hours):
            days_since_clean = (i - last_cleaning) / 24

            # Fouling groeit (biologisch + scaling + particulate)
            bio_fouling = 0.001 * days_since_clean ** 1.2
            scaling = 0.0005 * days_since_clean * (feed_tds[i] / 500) ** 2
            particulate = 0.0002 * days_since_clean * feed_turbidity[i]

            fouling_resistance[i] = bio_fouling + scaling + particulate + np.random.normal(0, 0.0005)
            fouling_resistance[i] = max(0, fouling_resistance[i])

            # Reiniging elke ~30 dagen of als TMP te hoog
            tmp_current = feed_pressure[i] * (1 + fouling_resistance[i] * 100)
            max_tmp = {"RO": 25, "UF": 3.5, "NF": 14}[module_type]

            if days_since_clean > 25 + np.random.randint(-5, 5) or tmp_current > max_tmp * 0.95:
                fouling_resistance[i] = fouling_resistance[i] * np.random.uniform(0.1, 0.3)  # niet perfect schoon
                last_cleaning = i
                cleaning_events.append(i)

        # Transmembraandruk
        tmp = feed_pressure * (1 + fouling_resistance * 100) + np.random.normal(0, 0.05, n_hours)

        # Permeaat flux (L/m²/h)
        base_flux = {"RO": 25, "UF": 80, "NF": 40}[module_type]
        # Temperatuur correctie (hoger temp = hogere flux)
        temp_factor = np.exp(0.03 * (feed_temp - 20))
        flux = base_flux * temp_factor / (1 + fouling_resistance * 200) + np.random.normal(0, 0.5, n_hours)
        flux = np.clip(flux, 5, 150)

        # Specifieke flux (flux / TMP)
        specific_flux = flux / tmp

        # Permeaat kwaliteit
        rejection = {"RO": 0.995, "UF": 0.5, "NF": 0.85}[module_type]
        rejection_actual = rejection - 0.01 * fouling_resistance * 50 + np.random.normal(0, 0.002, n_hours)
        permeate_tds = feed_tds * (1 - rejection_actual)
        permeate_conductivity = permeate_tds * 0.002 + np.random.normal(0, 0.02, n_hours)

        # Recovery
        recovery = np.random.uniform(0.7, 0.85, n_hours) + np.random.normal(0, 0.01, n_hours)

        # Energieverbruik
        sec = tmp * 100 / (flux * 36)  # kWh/m³ (vereenvoudigd)
        sec += np.random.normal(0, 0.05, n_hours)

        # Event labels
        is_cleaning = np.zeros(n_hours, dtype=int)
        for ce in cleaning_events:
            is_cleaning[max(0, ce-2):min(n_hours, ce+6)] = 1

        # Fouling fase
        fouling_phase = np.full(n_hours, "normaal", dtype=object)
        for i in range(n_hours):
            if is_cleaning[i]:
                fouling_phase[i] = "reiniging"
            elif fouling_resistance[i] > 0.03:
                fouling_phase[i] = "ernstige_fouling"
            elif fouling_resistance[i] > 0.015:
                fouling_phase[i] = "matige_fouling"
            elif fouling_resistance[i] > 0.005:
                fouling_phase[i] = "lichte_fouling"
            else:
                fouling_phase[i] = "schoon"

        for i in range(n_hours):
            rows.append({
                "timestamp": timestamps[i].strftime("%Y-%m-%d %H:%M"),
                "module_id": f"MEM-{module_id+1:02d}",
                "module_type": module_type,
                "voeding_TDS_mgL": round(feed_tds[i], 0),
                "voeding_turbiditeit_NTU": round(feed_turbidity[i], 2),
                "voeding_temp_C": round(feed_temp[i], 1),
                "voeding_pH": round(feed_ph[i], 1),
                "voeding_druk_bar": round(feed_pressure[i], 2),
                "TMP_bar": round(tmp[i], 2),
                "flux_Lm2h": round(flux[i], 1),
                "specifieke_flux_Lm2hbar": round(specific_flux[i], 2),
                "permeaat_TDS_mgL": round(max(0, permeate_tds[i]), 1),
                "permeaat_conductiviteit_mScm": round(max(0, permeate_conductivity[i]), 3),
                "recovery": round(recovery[i], 3),
                "fouling_weerstand": round(fouling_resistance[i], 5),
                "SEC_kWhm3": round(max(0, sec[i]), 3),
                "is_reiniging": is_cleaning[i],
                "fouling_fase": fouling_phase[i],
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/membraan_filtratie.csv", index=False)
    print(f"membraan_filtratie.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_sensor_fusion():
    """
    Dataset 41: Sensor Fusion - Meerdere sensoren meten dezelfde procesgrootheid
    met verschillende nauwkeurigheden, sampletijden en faalmodi.
    Doel: Sensor fusion, sensor selectie, fault-tolerant estimation, redundantie-analyse.
    """
    n_hours = 60 * 24  # 60 dagen, per uur (basistijd)
    n = n_hours
    t = np.arange(n)
    timestamps = [datetime(2025, 10, 1) + timedelta(hours=int(h)) for h in t]

    # --- Werkelijke proceswaarden (ground truth) ---
    true_temp = 150 + 10 * np.sin(2 * np.pi * t / (24 * 7)) + 3 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, n)
    true_pressure = 5.0 + 0.5 * np.sin(2 * np.pi * t / (24 * 3)) + np.random.normal(0, 0.01, n)
    true_flow = 100 + 15 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, n)
    true_level = 60 + 10 * np.sin(2 * np.pi * t / (24 * 2)) + np.random.normal(0, 0.2, n)

    # --- TEMPERATUUR: 4 sensoren (thermokoppel, RTD, IR, inline) ---
    # Sensor T1: Thermokoppel K-type (snel, medium nauwkeurig)
    t1 = true_temp + np.random.normal(0, 1.5, n)
    t1_status = np.full(n, "ok", dtype=object)
    # Drift na dag 30
    t1[30*24:] += np.linspace(0, 4, n - 30*24)
    t1_status[45*24:] = "drift"
    # Uitval dag 50-51
    t1[50*24:51*24] = np.nan
    t1_status[50*24:51*24] = "uitval"

    # Sensor T2: RTD Pt100 (traag, zeer nauwkeurig)
    t2_delay = 3  # 3 uur vertraging
    t2 = np.roll(true_temp, t2_delay) + np.random.normal(0, 0.3, n)
    t2[:t2_delay] = true_temp[:t2_delay] + np.random.normal(0, 0.3, t2_delay)
    t2_status = np.full(n, "ok", dtype=object)

    # Sensor T3: IR pyrometer (snel, maar ruis bij stoom/vocht)
    t3 = true_temp + np.random.normal(0, 0.8, n)
    # Periodieke verstoring door stoom
    steam_events = np.random.random(n) < 0.05
    t3[steam_events] += np.random.uniform(5, 20, steam_events.sum())
    t3_status = np.where(steam_events, "stoom_interferentie", "ok")

    # Sensor T4: Inline thermistor (goedkoop, beperkt bereik, niet-lineair)
    t4 = true_temp + 0.01 * (true_temp - 150) ** 2 + np.random.normal(0, 2, n)  # niet-lineaire fout
    t4_status = np.full(n, "ok", dtype=object)
    # Saturatie boven 165°C
    t4_saturated = t4 > 165
    t4[t4_saturated] = 165 + np.random.normal(0, 0.5, t4_saturated.sum())
    t4_status[t4_saturated] = "saturatie"

    # --- DRUK: 3 sensoren ---
    p1 = true_pressure + np.random.normal(0, 0.05, n)  # Capacitief (nauwkeurig)
    p1_status = np.full(n, "ok", dtype=object)

    p2 = true_pressure + np.random.normal(0, 0.1, n)  # Piezo-resistief
    p2_status = np.full(n, "ok", dtype=object)
    # Bevriezing (impulsleiding) dag 20-22
    p2[20*24:22*24] = p2[20*24]  # bevroren waarde
    p2_status[20*24:22*24] = "bevroren"

    p3 = true_pressure + np.random.normal(0, 0.15, n)  # Bourdon (oud, minder nauwkeurig)
    p3_status = np.full(n, "ok", dtype=object)
    # Hysterese
    for i in range(1, n):
        if abs(true_pressure[i] - true_pressure[i-1]) > 0.1:
            p3[i] += 0.05 * np.sign(true_pressure[i] - true_pressure[i-1])

    # --- DEBIET: 3 sensoren ---
    f1 = true_flow + np.random.normal(0, 1, n)  # Coriolis (zeer nauwkeurig)
    f1_status = np.full(n, "ok", dtype=object)

    f2 = true_flow + np.random.normal(0, 3, n)  # Vortex
    f2_status = np.full(n, "ok", dtype=object)
    # Slecht bij laag debiet
    low_flow = true_flow < 90
    f2[low_flow] += np.random.normal(0, 8, low_flow.sum())
    f2_status[low_flow] = "laag_debiet_onbetrouwbaar"

    f3 = true_flow * (1 + 0.001 * t / 24) + np.random.normal(0, 2, n)  # DP (kalibratie drift)
    f3_status = np.full(n, "ok", dtype=object)
    f3_status[40*24:] = "kalibratie_nodig"

    # --- NIVEAU: 2 sensoren ---
    l1 = true_level + np.random.normal(0, 0.5, n)  # Radar
    l1_status = np.full(n, "ok", dtype=object)
    # Schuim interferentie
    foam = np.random.random(n) < 0.03
    l1[foam] += np.random.uniform(-5, 10, foam.sum())
    l1_status[foam] = "schuim"

    l2 = true_level + np.random.normal(0, 1, n)  # DP niveaumeting
    l2_status = np.full(n, "ok", dtype=object)

    # --- Beste schatting (gewogen gemiddelde als referentie) ---
    # Temperatuur: gewogen naar nauwkeurigheid (negeer NaN/uitval)
    best_temp = np.nanmean([
        np.where(np.isnan(t1), np.nan, t1),
        t2, t3, t4
    ], axis=0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        # Ground truth
        "temp_werkelijk_C": np.round(true_temp, 2),
        "druk_werkelijk_bar": np.round(true_pressure, 3),
        "debiet_werkelijk_kgh": np.round(true_flow, 1),
        "niveau_werkelijk_pct": np.round(true_level, 1),
        # Temperatuur sensoren
        "T1_thermokoppel_C": [round(v, 1) if not np.isnan(v) else None for v in t1],
        "T1_status": t1_status,
        "T2_RTD_C": np.round(t2, 2),
        "T2_status": t2_status,
        "T3_IR_C": np.round(t3, 1),
        "T3_status": t3_status,
        "T4_thermistor_C": np.round(t4, 1),
        "T4_status": t4_status,
        # Druk sensoren
        "P1_capacitief_bar": np.round(p1, 3),
        "P1_status": p1_status,
        "P2_piezo_bar": np.round(p2, 3),
        "P2_status": p2_status,
        "P3_bourdon_bar": np.round(p3, 3),
        "P3_status": p3_status,
        # Debiet sensoren
        "F1_coriolis_kgh": np.round(f1, 1),
        "F1_status": f1_status,
        "F2_vortex_kgh": np.round(f2, 1),
        "F2_status": f2_status,
        "F3_DP_kgh": np.round(f3, 1),
        "F3_status": f3_status,
        # Niveau sensoren
        "L1_radar_pct": np.round(l1, 1),
        "L1_status": l1_status,
        "L2_DP_pct": np.round(l2, 1),
        "L2_status": l2_status,
    })

    df.to_csv("data/sensor_fusion.csv", index=False)
    print(f"sensor_fusion.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_concept_drift_longterm():
    """
    Dataset 42: Concept Drift over meerdere jaren - Productieproces met geleidelijke
    en abrupte veranderingen over 5 jaar. Model veroudering simulatie.
    Doel: Concept drift detectie, adaptief modelleren, model monitoring, retraining strategie.
    """
    n_days = 5 * 365  # 5 jaar
    n = n_days * 24  # per uur
    t = np.arange(n)
    timestamps = [datetime(2020, 1, 1) + timedelta(hours=int(h)) for h in t]

    # Seizoenspatroon (jaarlijks)
    day_of_year = np.array([(datetime(2020, 1, 1) + timedelta(hours=int(h))).timetuple().tm_yday for h in t])
    season = np.sin(2 * np.pi * day_of_year / 365)

    # Dag/nacht
    hour = np.array([(datetime(2020, 1, 1) + timedelta(hours=int(h))).hour for h in t])
    daynight = np.sin(2 * np.pi * hour / 24)

    # Input features (langzaam veranderend)
    feed_quality = 0.95 + 0.02 * season + np.random.normal(0, 0.005, n)
    temperature = 150 + 5 * season + 2 * daynight + np.random.normal(0, 0.5, n)
    pressure = 5 + 0.3 * season + np.random.normal(0, 0.05, n)
    flow = 100 + 10 * daynight + np.random.normal(0, 1, n)
    catalyst_age = np.zeros(n)
    ambient_temp = 15 + 12 * season + 5 * daynight + np.random.normal(0, 1, n)

    # --- Drift mechanismen ---

    # 1. Geleidelijke drift: katalysator veroudering (elke 6 maanden vervangen)
    replacement_interval = 180 * 24  # uur
    for i in range(n):
        catalyst_age[i] = (i % replacement_interval) / 24  # dagen

    catalyst_factor = 1 - 0.0005 * catalyst_age  # geleidelijke degradatie

    # 2. Abrupte verandering: nieuwe grondstof leverancier (jaar 2, maand 3)
    supplier_change = 2 * 365 * 24 + 3 * 30 * 24
    supplier_effect = np.zeros(n)
    supplier_effect[supplier_change:] = 0.03  # licht hogere conversie

    # 3. Geleidelijke verandering: fouling in warmtewisselaar
    fouling_buildup = np.zeros(n)
    last_clean = 0
    for i in range(n):
        days_since = (i - last_clean) / 24
        fouling_buildup[i] = 0.001 * days_since ** 1.2
        # Schoonmaak elke 90 dagen
        if days_since > 90:
            fouling_buildup[i] = 0
            last_clean = i

    # 4. Abrupte verandering: equipment upgrade (jaar 3, maand 6)
    upgrade_time = 3 * 365 * 24 + 6 * 30 * 24
    upgrade_effect = np.zeros(n)
    upgrade_effect[upgrade_time:] = -0.02  # betere efficiency, minder energieverbruik

    # 5. Regelgeving verandering: lagere emissienorm (jaar 4)
    regulation_time = 4 * 365 * 24
    regulation_effect = np.zeros(n)
    regulation_effect[regulation_time:] = 0.01  # operatie aangepast

    # Target: conversie
    conversion = (
        0.85 * catalyst_factor
        + supplier_effect
        + 0.002 * (temperature - 150)
        - 0.001 * (flow - 100)
        + 0.01 * (feed_quality - 0.95) * 10
        - fouling_buildup * 5
        + upgrade_effect
        + regulation_effect
        + np.random.normal(0, 0.005, n)
    )
    conversion = np.clip(conversion, 0.5, 0.99)

    # Target: energieverbruik
    energy = (
        50
        + 0.2 * flow
        + 0.1 * temperature
        + 10 * fouling_buildup * 100
        - 5 * np.where(np.arange(n) >= upgrade_time, 1, 0)  # na upgrade lager
        + 3 * np.maximum(0, ambient_temp - 25) / 10
        + np.random.normal(0, 1, n)
    )

    # Drift labels
    drift_type = np.full(n, "geen", dtype=object)

    # Geleidelijke drifts
    for i in range(n):
        if catalyst_age[i] > 120:
            drift_type[i] = "katalysator_veroudering"
        if fouling_buildup[i] > 0.003:
            drift_type[i] = "fouling"

    # Abrupte drifts (window rond event)
    window = 7 * 24
    drift_type[supplier_change:supplier_change+window] = "leverancier_wissel"
    drift_type[upgrade_time:upgrade_time+window] = "equipment_upgrade"
    drift_type[regulation_time:regulation_time+window] = "regelgeving_aanpassing"

    # Jaar en kwartaal (voor analyse)
    year = np.array([(datetime(2020, 1, 1) + timedelta(hours=int(h))).year for h in t])
    quarter = np.array([f"Q{((datetime(2020, 1, 1) + timedelta(hours=int(h))).month - 1) // 3 + 1}" for h in t])

    # Model performance indicatoren (gesimuleerd)
    # Een model getraind op jaar 1 data
    model_baseline_pred = (
        0.85
        + 0.002 * (temperature - 150)
        - 0.001 * (flow - 100)
        + 0.01 * (feed_quality - 0.95) * 10
        + np.random.normal(0, 0.005, n)
    )
    model_error = conversion - model_baseline_pred
    model_mae = np.abs(model_error)

    # Downsample naar elke 4 uur voor bestandsgrootte
    idx = np.arange(0, n, 4)
    df = pd.DataFrame({
        "timestamp": [timestamps[i] for i in idx],
        "jaar": year[idx],
        "kwartaal": quarter[idx],
        "voeding_kwaliteit": np.round(feed_quality[idx], 4),
        "temperatuur_C": np.round(temperature[idx], 1),
        "druk_bar": np.round(pressure[idx], 2),
        "debiet_kgh": np.round(flow[idx], 1),
        "omgevingstemp_C": np.round(ambient_temp[idx], 1),
        "katalysator_leeftijd_dagen": np.round(catalyst_age[idx], 0).astype(int),
        "fouling_index": np.round(fouling_buildup[idx], 5),
        "conversie": np.round(conversion[idx], 4),
        "energieverbruik_kWh": np.round(energy[idx], 1),
        "model_voorspelling": np.round(model_baseline_pred[idx], 4),
        "model_fout": np.round(model_error[idx], 5),
        "model_abs_fout": np.round(model_mae[idx], 5),
        "drift_type": drift_type[idx],
    })

    df.to_csv("data/concept_drift_5jaar.csv", index=False)
    print(f"concept_drift_5jaar.csv: {len(df)} rijen, {len(df.columns)} kolommen")


def generate_causal_inference():
    """
    Dataset 43: Causal Inference - Chemisch proces met bekende causale structuur (DAG).
    Inclusief interventies, confounders, en mediators.
    Doel: Causal discovery, causal effect estimation, do-calculus, counterfactuals.
    """
    n = 5000

    # Bekende causale structuur (DAG):
    # omgeving -> koelwater_temp -> reactor_temp -> conversie -> opbrengst
    # omgeving -> luchtvochtigheid -> vochtgehalte_product
    # katalysator_type -> activeringsenergie -> reactor_temp -> conversie
    # katalysator_type -> selectiviteit -> opbrengst
    # druk -> reactor_temp (confounder)
    # druk -> conversie
    # roersnelheid -> menging -> conversie
    # roersnelheid -> energieverbruik
    # operator -> roersnelheid (confounder: ervaren operators kiezen betere snelheid)

    # Exogene variabelen
    omgevingstemp = np.random.normal(20, 8, n)  # seizoensgebonden
    operator_ervaring = np.random.uniform(0, 20, n)  # jaren
    katalysator_type = np.random.choice(["Pd/C", "Pt/Al2O3", "Ni/SiO2"], n, p=[0.4, 0.35, 0.25])

    # Causale keten
    # Operator -> roersnelheid (ervaren operators kiezen optimaler)
    roersnelheid = 200 + 5 * operator_ervaring + np.random.normal(0, 20, n)
    roersnelheid = np.clip(roersnelheid, 50, 500)

    # Omgeving -> koelwater temp
    koelwater_temp = 15 + 0.4 * omgevingstemp + np.random.normal(0, 1, n)

    # Omgeving -> luchtvochtigheid
    luchtvochtigheid = 50 + 0.8 * omgevingstemp + np.random.normal(0, 5, n)
    luchtvochtigheid = np.clip(luchtvochtigheid, 20, 95)

    # Druk (onafhankelijk instelbaar, maar beinvloedt meerdere dingen)
    druk = np.random.uniform(1, 10, n)

    # Katalysator -> activeringsenergie
    ea_map = {"Pd/C": 50, "Pt/Al2O3": 45, "Ni/SiO2": 60}
    activeringsenergie = np.array([ea_map[k] for k in katalysator_type]) + np.random.normal(0, 2, n)

    # Reactor temperatuur (beïnvloed door koelwater, druk, activeringsenergie)
    reactor_temp = (
        150
        + 0.5 * (koelwater_temp - 15)
        + 2 * (druk - 5)
        - 0.3 * (activeringsenergie - 50)
        + np.random.normal(0, 2, n)
    )

    # Menging (beïnvloed door roersnelheid, niet-lineair)
    menging = 0.5 + 0.4 * (1 - np.exp(-roersnelheid / 200)) + np.random.normal(0, 0.03, n)
    menging = np.clip(menging, 0.1, 1.0)

    # Katalysator -> selectiviteit
    sel_map = {"Pd/C": 0.90, "Pt/Al2O3": 0.85, "Ni/SiO2": 0.80}
    selectiviteit = np.array([sel_map[k] for k in katalysator_type])
    selectiviteit += 0.01 * (reactor_temp - 150) / 10 + np.random.normal(0, 0.02, n)
    selectiviteit = np.clip(selectiviteit, 0.5, 0.99)

    # Conversie (beïnvloed door temp, druk, menging, activeringsenergie)
    conversie = (
        0.6
        + 0.003 * (reactor_temp - 150)
        + 0.01 * (druk - 5)
        + 0.15 * menging
        - 0.002 * (activeringsenergie - 50)
        + np.random.normal(0, 0.02, n)
    )
    conversie = np.clip(conversie, 0.1, 0.99)

    # Opbrengst (conversie * selectiviteit)
    opbrengst = conversie * selectiviteit * 100 + np.random.normal(0, 1, n)
    opbrengst = np.clip(opbrengst, 5, 99)

    # Energieverbruik (roersnelheid + temperatuur)
    energieverbruik = 10 + 0.05 * roersnelheid + 0.2 * reactor_temp + np.random.normal(0, 2, n)

    # Productvochtgehalte (luchtvochtigheid + temperatuur)
    vochtgehalte_product = 2 + 0.02 * luchtvochtigheid - 0.01 * reactor_temp + np.random.normal(0, 0.3, n)
    vochtgehalte_product = np.clip(vochtgehalte_product, 0.1, 8)

    # Interventies (sommige rijen zijn experimenten waar een variabele geforceerd is)
    interventie = np.full(n, "observatie", dtype=object)

    # do(roersnelheid=300) experimenten
    interv_idx = np.random.choice(n, 200, replace=False)
    interventie[interv_idx] = "do(roersnelheid=300)"
    roersnelheid[interv_idx] = 300

    # do(druk=8) experimenten
    interv_idx2 = np.random.choice(np.setdiff1d(np.arange(n), interv_idx), 200, replace=False)
    interventie[interv_idx2] = "do(druk=8)"
    druk[interv_idx2] = 8

    # Recalculeer na interventies
    for idx in np.concatenate([interv_idx, interv_idx2]):
        menging[idx] = 0.5 + 0.4 * (1 - np.exp(-roersnelheid[idx] / 200)) + np.random.normal(0, 0.03)
        reactor_temp[idx] = 150 + 0.5 * (koelwater_temp[idx] - 15) + 2 * (druk[idx] - 5) - 0.3 * (activeringsenergie[idx] - 50) + np.random.normal(0, 2)
        conversie[idx] = 0.6 + 0.003 * (reactor_temp[idx] - 150) + 0.01 * (druk[idx] - 5) + 0.15 * menging[idx] - 0.002 * (activeringsenergie[idx] - 50) + np.random.normal(0, 0.02)
        conversie[idx] = np.clip(conversie[idx], 0.1, 0.99)
        opbrengst[idx] = conversie[idx] * selectiviteit[idx] * 100 + np.random.normal(0, 1)

    df = pd.DataFrame({
        "sample_id": [f"CI-{i:05d}" for i in range(n)],
        "interventie": interventie,
        # Exogeen
        "omgevingstemp_C": np.round(omgevingstemp, 1),
        "operator_ervaring_jaar": np.round(operator_ervaring, 1),
        "katalysator_type": katalysator_type,
        # Causale variabelen
        "roersnelheid_RPM": np.round(roersnelheid, 0).astype(int),
        "koelwater_temp_C": np.round(koelwater_temp, 1),
        "luchtvochtigheid_pct": np.round(luchtvochtigheid, 1),
        "druk_bar": np.round(druk, 2),
        "activeringsenergie_kJmol": np.round(activeringsenergie, 1),
        "reactor_temp_C": np.round(reactor_temp, 1),
        "menging_index": np.round(menging, 3),
        "selectiviteit": np.round(selectiviteit, 3),
        "conversie": np.round(conversie, 4),
        "opbrengst_pct": np.round(opbrengst, 1),
        "energieverbruik_kWh": np.round(energieverbruik, 1),
        "vochtgehalte_product_pct": np.round(vochtgehalte_product, 2),
    })

    # Voeg bekende DAG toe als metadata in apart bestand
    dag_edges = [
        ("omgevingstemp_C", "koelwater_temp_C"),
        ("omgevingstemp_C", "luchtvochtigheid_pct"),
        ("koelwater_temp_C", "reactor_temp_C"),
        ("druk_bar", "reactor_temp_C"),
        ("druk_bar", "conversie"),
        ("activeringsenergie_kJmol", "reactor_temp_C"),
        ("katalysator_type", "activeringsenergie_kJmol"),
        ("katalysator_type", "selectiviteit"),
        ("reactor_temp_C", "conversie"),
        ("reactor_temp_C", "selectiviteit"),
        ("operator_ervaring_jaar", "roersnelheid_RPM"),
        ("roersnelheid_RPM", "menging_index"),
        ("roersnelheid_RPM", "energieverbruik_kWh"),
        ("menging_index", "conversie"),
        ("conversie", "opbrengst_pct"),
        ("selectiviteit", "opbrengst_pct"),
        ("luchtvochtigheid_pct", "vochtgehalte_product_pct"),
        ("reactor_temp_C", "vochtgehalte_product_pct"),
        ("reactor_temp_C", "energieverbruik_kWh"),
    ]
    dag_df = pd.DataFrame(dag_edges, columns=["oorzaak", "gevolg"])
    dag_df.to_csv("data/causal_dag_structuur.csv", index=False)

    df.to_csv("data/causal_inference_proces.csv", index=False)
    print(f"causal_inference_proces.csv: {len(df)} rijen, {len(df.columns)} kolommen")
    print(f"causal_dag_structuur.csv: {len(dag_df)} rijen (bekende DAG edges)")


def generate_federated_learning():
    """
    Dataset 44: Federated Learning - 4 farmaceutische productiesites met dezelfde
    productielijn maar lokale variaties. Data mag niet gecombineerd worden (privacy).
    Doel: Federated learning, privacy-preserving ML, model aggregatie.
    """
    sites = {
        "Site_NL": {"n": 500, "temp_offset": 0, "humidity": 45, "skill_level": 0.9, "equipment_age": 2},
        "Site_DE": {"n": 400, "temp_offset": -2, "humidity": 40, "skill_level": 0.85, "equipment_age": 5},
        "Site_US": {"n": 600, "temp_offset": 3, "humidity": 55, "skill_level": 0.8, "equipment_age": 1},
        "Site_IN": {"n": 300, "temp_offset": 8, "humidity": 70, "skill_level": 0.75, "equipment_age": 8},
    }

    all_rows = []
    for site_name, props in sites.items():
        n = props["n"]

        # Procesparameters (lokaal gemeten)
        temp = np.random.normal(150 + props["temp_offset"], 3, n)
        pressure = np.random.normal(5, 0.3, n)
        flow = np.random.normal(100, 5, n)
        humidity = np.random.normal(props["humidity"], 3, n)
        stir_speed = np.random.normal(250 * props["skill_level"] / 0.85, 20, n)

        # Grondstof (lokale leveranciers)
        api_purity = np.random.normal(99.5, 0.2, n)
        excipient_moisture = np.random.normal(2 + props["humidity"] * 0.01, 0.3, n)

        # Equipment effect (ouder = meer variatie)
        equipment_noise = 0.5 * props["equipment_age"] / 5

        # Target: productkwaliteit
        quality = (
            85
            + 0.3 * (temp - 150)
            - 2 * (humidity - 50) / 10
            + 0.1 * stir_speed / 10
            + 1.5 * (api_purity - 99.5)
            - 3 * (excipient_moisture - 2)
            + 0.5 * pressure
            - 0.01 * flow
            + np.random.normal(0, 1 + equipment_noise, n)
        )
        quality = np.clip(quality, 50, 100)

        # Dissolution
        dissolution = (
            90
            - 0.5 * (quality - 85) * 0.3
            + 0.1 * (temp - 150)
            + np.random.normal(0, 2, n)
        )
        dissolution = np.clip(dissolution, 50, 100)

        # Goedkeuring
        approved = ((quality > 75) & (dissolution > 75)).astype(int)

        # Data verdeling per site (niet-IID: verschillende distributies)
        for i in range(n):
            all_rows.append({
                "sample_id": f"{site_name}-{i:04d}",
                "site": site_name,
                "temperatuur_C": round(temp[i], 1),
                "druk_bar": round(pressure[i], 2),
                "debiet_kgh": round(flow[i], 1),
                "luchtvochtigheid_pct": round(humidity[i], 1),
                "roersnelheid_RPM": round(stir_speed[i], 0),
                "API_zuiverheid_pct": round(api_purity[i], 2),
                "excipient_vocht_pct": round(excipient_moisture[i], 2),
                "equipment_leeftijd_jaar": props["equipment_age"],
                "kwaliteitsscore": round(quality[i], 1),
                "dissolutie_pct": round(dissolution[i], 1),
                "goedgekeurd": approved[i],
            })

    df = pd.DataFrame(all_rows)
    df.to_csv("data/federated_learning_sites.csv", index=False)
    print(f"federated_learning_sites.csv: {len(df)} rijen, {len(df.columns)} kolommen")

    # Ook per-site bestanden voor echte FL simulatie
    for site_name in sites:
        site_df = df[df["site"] == site_name].copy()
        site_df.to_csv(f"data/fl_site_{site_name.lower()}.csv", index=False)


def generate_timeseries_foundation():
    """
    Dataset 45: Time-Series Foundation Model data - Lange multivariate tijdreeks
    met 50 kanalen van een complete chemische plant. 2 jaar, elke 5 minuten.
    Doel: Foundation model pre-training, forecasting, imputation, anomaly detection.
    """
    n = 2 * 365 * 24 * 12  # 2 jaar, elke 5 min
    t = np.arange(n)

    # Downsample timestamps genereren
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=int(i * 5)) for i in t]

    # Tijdfeatures
    hour = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])

    # Seizoens- en dagcycli
    yearly = np.sin(2 * np.pi * day_of_year / 365)
    daily = np.sin(2 * np.pi * hour / 24)
    weekly = np.sin(2 * np.pi * day_of_week / 7)

    # --- 50 kanalen genereren ---
    channels = {}

    # Groep 1: Reactorsectie (10 kanalen)
    channels["R_temp_C"] = 180 + 5 * yearly + 2 * daily + np.random.normal(0, 0.5, n)
    channels["R_druk_bar"] = 8 + 0.3 * yearly + np.random.normal(0, 0.05, n)
    channels["R_niveau_pct"] = 60 + 5 * daily + np.random.normal(0, 1, n)
    channels["R_voeding_kgh"] = 100 + 10 * daily - 5 * (day_of_week >= 5).astype(float) + np.random.normal(0, 1, n)
    channels["R_conversie"] = 0.85 + 0.02 * yearly + 0.005 * daily + np.random.normal(0, 0.005, n)
    channels["R_roerder_RPM"] = 200 + np.random.normal(0, 2, n)
    channels["R_koelwater_in_C"] = 20 + 8 * yearly + 3 * daily + np.random.normal(0, 0.3, n)
    channels["R_koelwater_uit_C"] = channels["R_koelwater_in_C"] + 15 + 2 * (channels["R_temp_C"] - 180) / 5 + np.random.normal(0, 0.3, n)
    channels["R_pH"] = 7.0 + 0.2 * np.sin(2 * np.pi * t / (288 * 3)) + np.random.normal(0, 0.05, n)
    channels["R_viscositeit_mPas"] = 50 + 5 * yearly + np.random.normal(0, 1, n)

    # Groep 2: Scheidingssectie (10 kanalen)
    channels["S_kolom_top_C"] = 78 + 2 * yearly + np.random.normal(0, 0.3, n)
    channels["S_kolom_bodem_C"] = 120 + 3 * yearly + np.random.normal(0, 0.3, n)
    channels["S_reflux_ratio"] = 3.0 + 0.2 * np.sin(2 * np.pi * t / (288 * 7)) + np.random.normal(0, 0.05, n)
    channels["S_reboiler_kW"] = 200 + 20 * yearly + 10 * daily + np.random.normal(0, 3, n)
    channels["S_condenser_kW"] = -150 - 15 * yearly + np.random.normal(0, 2, n)
    channels["S_drukval_mbar"] = 50 + 5 * np.sin(2 * np.pi * t / (288 * 30)) + np.random.normal(0, 1, n)
    channels["S_product_zuiverheid"] = 0.995 + 0.002 * yearly + np.random.normal(0, 0.001, n)
    channels["S_product_flow_kgh"] = 80 + 8 * daily + np.random.normal(0, 1, n)
    channels["S_afval_flow_kgh"] = 20 + 2 * daily + np.random.normal(0, 0.5, n)
    channels["S_dampflow_kgh"] = 50 + 5 * daily + np.random.normal(0, 1, n)

    # Groep 3: Warmtewisseling & utiliteiten (10 kanalen)
    channels["U_stoom_tonh"] = 25 + 5 * yearly + 3 * daily + np.random.normal(0, 0.5, n)
    channels["U_stoom_druk_bar"] = 10 + np.random.normal(0, 0.1, n)
    channels["U_koelwater_m3h"] = 150 + 30 * yearly + 10 * daily + np.random.normal(0, 3, n)
    channels["U_koeltoren_C"] = 25 + 10 * yearly + np.random.normal(0, 0.5, n)
    channels["U_elektra_kW"] = 400 + 50 * daily - 30 * (day_of_week >= 5).astype(float) + np.random.normal(0, 10, n)
    channels["U_perslucht_bar"] = 7 + np.random.normal(0, 0.05, n)
    channels["U_N2_Nm3h"] = 30 + 5 * daily + np.random.normal(0, 1, n)
    channels["U_WKK_kW"] = 200 + 20 * daily + np.random.normal(0, 5, n)
    channels["U_buitentemp_C"] = 10 + 12 * yearly + 5 * daily + np.random.normal(0, 1, n)
    channels["U_windsnelheid_ms"] = 4 + 2 * np.abs(np.random.normal(0, 1, n))

    # Groep 4: Kwaliteit & lab (10 kanalen)
    channels["Q_product_pct"] = 99.5 + 0.1 * yearly + np.random.normal(0, 0.1, n)
    channels["Q_onzuiverheid_A_ppm"] = 50 - 5 * yearly + np.random.exponential(5, n)
    channels["Q_onzuiverheid_B_ppm"] = 20 + np.random.exponential(3, n)
    channels["Q_kleur_hazen"] = 10 + 3 * yearly + np.random.exponential(1, n)
    channels["Q_dichtheid_gcm3"] = 1.05 + 0.002 * yearly + np.random.normal(0, 0.001, n)
    channels["Q_watergehalte_ppm"] = 200 + 50 * yearly + np.random.exponential(20, n)
    channels["Q_zuurgraad_mgKOHg"] = 0.1 + np.random.exponential(0.02, n)
    channels["Q_viscositeit_cSt"] = 15 + 2 * yearly + np.random.normal(0, 0.3, n)
    channels["Q_smeltpunt_C"] = 65 + np.random.normal(0, 0.2, n)
    channels["Q_hardheid_shore"] = 80 + np.random.normal(0, 2, n)

    # Groep 5: Milieu & veiligheid (10 kanalen)
    channels["M_emissie_NOx_mgNm3"] = 80 + 10 * yearly + np.random.exponential(5, n)
    channels["M_emissie_SO2_mgNm3"] = 30 + 5 * yearly + np.random.exponential(3, n)
    channels["M_emissie_stof_mgNm3"] = 5 + np.random.exponential(1, n)
    channels["M_afvalwater_COD_mgL"] = 100 + 20 * daily + np.random.exponential(10, n)
    channels["M_afvalwater_pH"] = 7 + 0.3 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 0.1, n)
    channels["M_geluid_dBA"] = 75 + 3 * daily + np.random.normal(0, 1, n)
    channels["M_gasdetectie_LEL_pct"] = np.random.exponential(0.5, n)
    channels["M_brandmelders_actief"] = (np.random.random(n) < 0.001).astype(int)
    channels["M_noodstop_actief"] = (np.random.random(n) < 0.0002).astype(int)
    channels["M_productie_ton"] = 3 + 0.5 * daily - 0.3 * (day_of_week >= 5).astype(float) + np.random.normal(0, 0.1, n)

    # --- Injecteren van realistische patronen ---

    # Onderhoudsstops (4 per jaar, elk 5 dagen)
    for year_offset in range(2):
        for stop_day in [60, 150, 240, 330]:
            start = (year_offset * 365 + stop_day) * 288
            end = min(start + 5 * 288, n)
            if start < n:
                for key in channels:
                    if key.startswith(("R_", "S_")):
                        channels[key][start:end] *= 0.3

    # Seizoensgebonden anomalieën
    # Hittegolf zomer jaar 1
    heatwave_start = 180 * 288
    heatwave_end = heatwave_start + 10 * 288
    channels["U_buitentemp_C"][heatwave_start:heatwave_end] += 10
    channels["U_koelwater_m3h"][heatwave_start:heatwave_end] += 50
    channels["U_koeltoren_C"][heatwave_start:heatwave_end] += 5

    # Vorstperiode winter jaar 2
    frost_start = (365 + 30) * 288
    frost_end = frost_start + 7 * 288
    if frost_end < n:
        channels["U_buitentemp_C"][frost_start:frost_end] -= 15
        channels["U_stoom_tonh"][frost_start:frost_end] += 10

    # Missing data (random sensor uitval)
    for key in list(channels.keys())[:20]:
        n_missing = np.random.randint(50, 200)
        missing_starts = np.random.randint(0, n - 12, n_missing)
        for ms in missing_starts:
            duration = np.random.randint(1, 12)
            channels[key][ms:ms+duration] = np.nan

    # Clip alle kanalen
    for key in channels:
        channels[key] = np.clip(channels[key], -1000, 100000) if not np.isnan(channels[key]).all() else channels[key]

    # Rond af
    data = {"timestamp": timestamps}
    for key, values in channels.items():
        precision = 1 if "pct" in key or "C" in key or "kgh" in key else (3 if "zuiverheid" in key or "conversie" in key else 1)
        data[key] = [round(v, precision) if not np.isnan(v) else None for v in values]

    df = pd.DataFrame(data)

    # Downsample naar elke 15 min voor bestandsgrootte (nog steeds groot)
    df = df.iloc[::3].reset_index(drop=True)
    df.to_csv("data/timeseries_foundation_plant.csv", index=False)
    print(f"timeseries_foundation_plant.csv: {len(df)} rijen, {len(df.columns)} kolommen")


if __name__ == "__main__":
    print("=== Datasets genereren voor ML in Process Engineering ===\n")
    generate_batch_reactor()
    generate_distillation_column()
    generate_pharma_tablet_press()
    generate_cstr_experiment()
    generate_sensor_drift()
    generate_crystallization()
    generate_wwtp()
    generate_spc_controlchart()
    generate_incoming_qc()
    generate_fermentation()
    generate_hplc_stability()
    generate_heat_exchanger()
    generate_continuous_polymerization()
    generate_continuous_drying()
    generate_continuous_mixing()
    generate_multiunit_process()
    generate_compressor_monitoring()
    generate_cho_cell_culture()
    generate_chromatography()
    generate_lyophilization()
    generate_pid_control_loops()
    generate_alarm_management()
    generate_energy_optimization()
    generate_vision_inspection()
    generate_spectroscopy_nir()
    generate_valve_diagnostics()
    generate_cip_cleaning()
    generate_cleanroom_monitoring()
    generate_golden_batch()
    generate_mpc_process()
    generate_digital_twin()
    generate_operator_logs()
    generate_recipe_optimization()
    generate_rl_environment()
    generate_transfer_learning()
    generate_active_learning()
    generate_nlp_maintenance()
    generate_virtual_metrology()
    generate_extruder()
    generate_membrane_filtration()
    generate_sensor_fusion()
    generate_concept_drift_longterm()
    generate_causal_inference()
    generate_federated_learning()
    generate_timeseries_foundation()
    print("\nKlaar! Alle datasets staan in de 'data/' map.")

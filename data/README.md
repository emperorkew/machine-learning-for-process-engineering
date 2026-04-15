# Datasets - Machine Learning voor Process Engineering

Kunstmatige datasets gebaseerd op typische processen in de farmaceutische en chemische industrie.
Alle data wordt gegenereerd met `generate_datasets.py` (seed=42 voor reproduceerbaarheid).

---

## Overzicht

| # | Bestand | Rijen | Kolommen | Domein |
|---|---------|-------|----------|--------|
| 1 | `batch_reactor_yield.csv` | 1000 | 10 | Productie |
| 2 | `distillatiekolom_timeseries.csv` | 720 | 14 | Productie |
| 3 | `tabletpers_kwaliteit.csv` | 2000 | 14 | Kwaliteitscontrole |
| 4 | `cstr_experiment.csv` | 1500 | 11 | Productie |
| 5 | `sensor_drift_maintenance.csv` | 43200 | 8 | Onderhoud |
| 6 | `kristallisatie_proces.csv` | 800 | 13 | Productie |
| 7 | `afvalwater_behandeling.csv` | 2160 | 15 | Milieu/Proces |
| 8 | `spc_controlekaart.csv` | 2000 | 8 | Kwaliteitscontrole |
| 9 | `ingangscontrole_grondstoffen.csv` | 1200 | 14 | Kwaliteitscontrole |
| 10 | `fermentatie_bioreaktor.csv` | 5000 | 12 | Productie |
| 11 | `hplc_stabiliteit.csv` | 900 | 14 | Kwaliteitscontrole |
| 12 | `warmtewisselaar_fouling.csv` | 8760 | 14 | Onderhoud/Proces |
| 13 | `continue_polymerisatie.csv` | 20160 | 20 | Continue productie |
| 14 | `continue_droging.csv` | 18000 | 16 | Continue productie |
| 15 | `continue_menging.csv` | 17280 | 14 | Continue productie |
| 16 | `geintegreerde_productielijn.csv` | 17280 | 25 | Continue productie |
| 17 | `compressor_monitoring.csv` | 129600 | 20 | Continue monitoring |
| 18 | `cho_celcultuur_perfusie.csv` | 1440 | 26 | Biofarmaceutisch |
| 19 | `chromatografie_zuivering.csv` | 200 | 30 | Biofarmaceutisch |
| 20 | `lyofilisatie_cyclus.csv` | 118672 | 13 | Biofarmaceutisch |
| 21 | `pid_regelkringen.csv` | 57600 | 9 | Procesautomatisering |
| 22 | `alarmbeheer_logboek.csv` | 25052 | 12 | Procesautomatisering |
| 23 | `energie_utiliteiten.csv` | 35040 | 19 | Utiliteiten/Energie |
| 24 | `machine_vision_inspectie.csv` | 5000 | 22 | AI Kwaliteitscontrole |
| 25 | `nir_spectroscopie.csv` | 500 | 262 | AI Kwaliteitscontrole |
| 26 | `klep_diagnostiek.csv` | 1000 | 21 | Procesautomatisering |
| 27 | `cip_reiniging.csv` | 27000 | 17 | Procescontrole/GMP |
| 28 | `cleanroom_monitoring.csv` | 25920 | 14 | GMP Monitoring |
| 29 | `golden_batch_coating.csv` | 9600 | 15 | Batchanalyse |
| 30 | `mpc_destillatie.csv` | 2880 | 19 | Procesautomatisering |
| 31 | `digital_twin_validatie.csv` | 4320 | 18 | Industry 4.0 |
| 32 | `operator_logboeken.csv` | 1095 | 14 | Industry 4.0 / NLP |
| 33 | `recept_optimalisatie.csv` | 800 | 29 | Industry 4.0 |
| 34 | `rl_reactor_control.csv` | 1668 | 19 | Reinforcement Learning |
| 35 | `transfer_learning_reactoren.csv` | 2100 | 12 | Transfer Learning |
| 36 | `active_learning_pool.csv` | 5000 | 16 | Active Learning |
| 37 | `nlp_onderhoudslogboek.csv` | 3000 | 17 | NLP / Text Mining |
| 38 | `virtual_metrology.csv` | 3000 | 20 | Semi-supervised ML |
| 39 | `extrusie_hotmelt.csv` | 17280 | 28 | Continue productie |
| 40 | `membraan_filtratie.csv` | 17280 | 18 | Procesmonitoring |

---

## Gedetailleerde beschrijving per dataset

### 1. Batch Reactor Yield (`batch_reactor_yield.csv`)

**Context:** Exotherme reactie in een farmaceutische batch reactor. Elke rij is een voltooide batch.

**Kolommen:**
- `batch_id` - unieke batch identifier
- `temperatuur_C` - reactietemperatuur (60-90 °C)
- `druk_bar` - procesdruk (gecorreleerd met temperatuur)
- `roersnelheid_RPM` - roerdersnelheid (100-500 RPM)
- `koelwater_temp_C` - temperatuur koelwater (5-25 °C)
- `reactietijd_min` - duur van de reactie (30-180 min)
- `katalysator_g` - hoeveelheid katalysator (0.5-5 g)
- `pH` - pH van het reactiemengsel (5-9)
- `opbrengst_pct` - **target (regressie):** eindopbrengst (%)
- `kwaliteit` - **target (classificatie):** Premium / Standaard / Afgekeurd

**ML-toepassingen:** Regressie (opbrengst voorspellen), multiclass classificatie, feature importance analyse, procesoptimalisatie.

---

### 2. Distillatiekolom Timeseries (`distillatiekolom_timeseries.csv`)

**Context:** Continue scheiding van een binair mengsel in een distillatiekolom met 5 trays. Uurlijkse metingen over 30 dagen.

**Kolommen:**
- `timestamp` - tijdstip van meting
- `feed_flow_kgh` - voedingsdebiet (kg/h)
- `feed_samenstelling_molfrac` - molfractie lichte component in voeding
- `feed_temp_C` - voedingstemperatuur (°C)
- `reflux_ratio` - terugvloeiverhouding
- `reboiler_duty_kW`, `condenser_duty_kW` - energieverbruik
- `tray_1_temp_C` t/m `tray_5_temp_C` - temperaturen op 5 trays
- `top_zuiverheid` - **target (regressie):** zuiverheid topproduct (molfractie)
- `anomalie` - **target (classificatie):** 1 = flooding/foaming event

**ML-toepassingen:** Tijdreeksanalyse, anomalie-detectie, soft sensor ontwikkeling, multivariate procesmonitoring.

**Bijzonderheden:** Bevat 4 geinjecteerde anomalieperioden (flooding/foaming) met verhoogde tray-temperaturen en verlaagde zuiverheid.

---

### 3. Tabletpers Kwaliteit (`tabletpers_kwaliteit.csv`)

**Context:** Kwaliteitscontrole van farmaceutische tabletten op een roterende tabletpers. Elke rij is een individueel tablet.

**Kolommen:**
- `tablet_id` - unieke tablet identifier
- `deeltjesgrootte_um` - deeltjesgrootte API (µm)
- `vochtgehalte_pct` - vochtgehalte grondstofmengsel (%)
- `menguniformiteit_pct` - uniformiteit van het poedermengsel (%)
- `perskracht_kN` - hoofdperskracht (kN)
- `voorcompressie_kN` - voorcompressiekracht (kN)
- `draaisnelheid_RPM` - toerensnelheid draaitafel (RPM)
- `vuldiepte_mm` - vuldiepte matrijs (mm)
- `gewicht_mg` - tabletgewicht (mg), spec: 350 ± 5%
- `dikte_mm` - tabletdikte (mm), spec: 3.5-4.5
- `hardheid_N` - **target (regressie):** breeksterkte (N), spec: 50-120
- `brosheid_pct` - afslijting (%), spec: < 1.0
- `dissolutie_30min_pct` - vrijgifte na 30 min (%), spec: > 75
- `goedgekeurd` - **target (classificatie):** 1 = voldoet aan alle specs

**ML-toepassingen:** Binaire classificatie, regressie, procesoptimalisatie, root cause analysis bij afkeur.

---

### 4. CSTR Experiment (`cstr_experiment.csv`)

**Context:** Continu geroerde tankreactor (CSTR) met meervoudige steady states. Experimenten bij verschillende operatiecondities.

**Kolommen:**
- `experiment_id` - unieke experiment identifier
- `inlet_temp_K` - inlaattemperatuur (K)
- `inlet_conc_molL` - inlaatconcentratie (mol/L)
- `inlet_flow_Lmin` - inlaatdebiet (L/min)
- `koelwater_flow_Lmin` - koelwaterdebiet (L/min)
- `reactor_volume_L` - reactorvolume (5, 10 of 20 L)
- `verblijftijd_min` - berekende verblijftijd (min)
- `outlet_temp_K` - **target (regressie):** uitlaattemperatuur (K)
- `outlet_conc_molL` - uitlaatconcentratie (mol/L)
- `conversie_pct` - **target (regressie):** conversiegraad (%)
- `operatieregime` - **target (clustering):** laag / midden / hoog

**ML-toepassingen:** Clustering (ontdek de 3 regimes zonder labels), regressie, classificatie, niet-lineaire systeemidentificatie.

---

### 5. Sensor Drift & Maintenance (`sensor_drift_maintenance.csv`)

**Context:** 10 pH-sensoren die geleidelijk degraderen over 180 dagen. Uurlijkse metingen.

**Kolommen:**
- `sensor_id` - sensor identifier (pH-00 t/m pH-09)
- `dag` - dag sinds installatie (0-179)
- `uur` - uur van de dag (0-23)
- `werkelijke_pH` - referentiewaarde (altijd 7.0)
- `gemeten_pH` - **input:** gemeten waarde (driftt over tijd)
- `afwijking` - verschil gemeten - werkelijk
- `ruis_std` - standaarddeviatie van de ruis (neemt toe)
- `status` - **target (classificatie):** ok / kalibratie_nodig / defect

**ML-toepassingen:** Predictive maintenance, tijdreeksclassificatie, remaining useful life (RUL) voorspelling, drift-detectie.

**Bijzonderheden:** Elke sensor heeft een unieke degradatiesnelheid en een random faalpunt (dag 90-180).

---

### 6. Kristallisatie Proces (`kristallisatie_proces.csv`)

**Context:** Farmaceutische API-kristallisatie. Experimenten met verschillende oplosmiddelen, koelprofielen en zaadkristallen.

**Kolommen:**
- `experiment_id` - unieke experiment identifier
- `start_temp_C`, `eind_temp_C` - temperatuurtraject (°C)
- `koelsnelheid_C_min` - koelsnelheid (°C/min)
- `concentratie_gL` - initiële concentratie (g/L)
- `oplosmiddel` - water / ethanol / methanol / aceton
- `zaadkristallen_g` - hoeveelheid zaadkristallen (g)
- `roersnelheid_RPM` - roersnelheid (RPM)
- `anti_solvent_rate_mLmin` - anti-solvent toevoegsnelheid (mL/min)
- `gem_kristalgrootte_um` - **target (regressie):** gemiddelde kristalgrootte (µm)
- `kristalgrootte_CV_pct` - **target (regressie):** spreiding kristalgrootte (CV%)
- `opbrengst_pct` - opbrengst (%)
- `polymorf` - **target (classificatie):** Form_A / Form_B (kristalvorm)

**ML-toepassingen:** Regressie (kristalgrootte), classificatie (polymorf), procesoptimalisatie, feature engineering met categorische variabelen.

---

### 7. Afvalwaterbehandeling (`afvalwater_behandeling.csv`)

**Context:** Afvalwaterzuiveringsinstallatie (RWZI) van een chemische/farmaceutische plant. Uurlijkse metingen over 90 dagen.

**Kolommen:**
- `timestamp` - tijdstip
- `influent_debiet_m3h` - inkomend debiet (m³/h)
- `influent_COD_mgL` - chemisch zuurstofverbruik influent (mg/L)
- `influent_pH`, `influent_TSS_mgL`, `influent_NH4_mgL` - invloedparameters
- `beluchting_m3h` - beluchtingsdebiet (m³/h)
- `slibrecirculatie_ratio` - slibrecirculatieverhouding
- `slibverblijftijd_dagen` - slibverblijftijd (dagen)
- `opgeloste_O2_mgL` - opgeloste zuurstof in beluchting (mg/L)
- `effluent_COD_mgL` - **target (regressie):** COD effluent (mg/L)
- `effluent_pH`, `effluent_TSS_mgL`, `effluent_NH4_mgL` - effluentkwaliteit
- `lozingsnorm_overtreding` - **target (classificatie):** 1 = normoverschrijding

**ML-toepassingen:** Multi-output regressie, classificatie, procesoptimalisatie (beluchting minimaliseren), dag/nacht patronen.

**Bijzonderheden:** Duidelijk dag/nacht-patroon in influent. Lozingsnormen: COD < 125, NH4 < 10, TSS < 35 mg/L.

---

### 8. SPC Controlekaart (`spc_controlekaart.csv`)

**Context:** Statistical Process Control data van een verpakkingslijn (vulgewicht). Bevat normale variatie en 6 types controlekaartpatronen.

**Kolommen:**
- `sample_nr` - volgnummer meting
- `subgroep` - subgroep van 5 metingen
- `meetwaarde_g` - **input:** gemeten vulgewicht (g)
- `doel_g` - doelwaarde (500 g)
- `UCL_g`, `LCL_g` - bovenste en onderste controlelimieten (3 sigma)
- `buiten_limieten` - 1 als meetwaarde buiten UCL/LCL
- `patroon` - **target (classificatie):** type afwijkingspatroon

**Patronen in de data:**
- `normaal` - normale procesvariatie
- `trend_omhoog` (samples 200-300) - geleidelijke stijging
- `shift_omhoog` (samples 500-580) - plotselinge niveauverschuiving
- `verhoogde_variatie` (samples 800-900) - toename procesvariatie
- `cyclisch` (samples 1100-1250) - periodiek patroon
- `trend_omlaag` (samples 1500-1600) - geleidelijke daling
- `stratificatie` (samples 1750-1850) - onnatuurlijk lage variatie

**ML-toepassingen:** Multiclass classificatie (patroonherkenning), anomalie-detectie, change-point detectie, sliding window features.

---

### 9. Ingangscontrole Grondstoffen (`ingangscontrole_grondstoffen.csv`)

**Context:** GMP-ingangscontrole van farmaceutische grondstoffen. Analytische testen per lot van 4 leveranciers.

**Kolommen:**
- `lot_id` - lot identifier
- `datum` - ontvangstdatum
- `leverancier` - Leverancier_A t/m D (elk met eigen kwaliteitsprofiel)
- `grondstof` - Paracetamol_API / Lactose / Magnesiumstearaat / Cellulose_MCC
- `zuiverheid_pct` - gehalte (%), spec: > 98 (hertest bij 98-99)
- `vochtgehalte_pct` - vochtgehalte (%), spec: < 4.0
- `deeltjesgrootte_d50_um`, `deeltjesgrootte_d90_um` - deeltjesgrootteverdeling (µm)
- `bulkdichtheid_gmL`, `tapdichtheid_gmL` - dichtheidstesten (g/mL)
- `carr_index_pct` - compressibiliteitsindex (%), maat voor vloei-eigenschappen
- `zware_metalen_ppm` - zware metalen (ppm), spec: < 20
- `microbieel_CFUg` - microbiologische telling (CFU/g), spec: < 500
- `besluit` - **target (classificatie):** goedgekeurd / hertest / afgekeurd

**ML-toepassingen:** Multiclass classificatie, leveranciersanalyse, anomalie-detectie per leverancier, feature importance.

**Bijzonderheden:** Leverancier_D heeft systematisch hogere afwijkingen. Ongebalanceerde klassen (meeste lots goedgekeurd).

---

### 10. Fermentatie Bioreaktor (`fermentatie_bioreaktor.csv`)

**Context:** Fed-batch fermentatie in een bioreaktor voor recombinant eiwitproductie. 50 batches, elk met ~100 tijdspunten (elke 2 uur, ~8 dagen).

**Kolommen:**
- `batch_id` - batch identifier
- `tijdstip_uur` - tijdstip in de batch (uur)
- `temp_setpoint_C`, `temp_actueel_C` - temperatuur setpoint en actueel (30/33/37 °C)
- `pH_setpoint`, `pH_actueel` - pH setpoint en actueel
- `biomassa_gL` - **target (regressie):** biomassaconcentratie (g/L)
- `glucose_gL` - glucoseconcentratie (g/L)
- `voedingssnelheid_gLh` - voedingsdebiet (g/L/h)
- `voedingsstrategie` - constant / exponentieel / DO_gestuurd
- `opgeloste_O2_pct` - opgeloste zuurstof (%)
- `product_gL` - **target (regressie):** productconcentratie (g/L)

**ML-toepassingen:** Tijdreeksregressie, batch trajectory modelling, vergelijking voedingsstrategieën, groeicurve fitting.

**Bijzonderheden:** Logistische groei met lag-fase. Drie voedingsstrategieën met verschillende effecten op eindopbrengst.

---

### 11. HPLC Stabiliteitsonderzoek (`hplc_stabiliteit.csv`)

**Context:** Stabiliteitsstudie volgens ICH-richtlijnen. 30 batches getest bij 3 opslagcondities over 36 maanden, 3 formuleringen.

**Kolommen:**
- `batch_id` - batch identifier
- `formulering` - tablet / capsule / suspensie
- `opslagconditie` - 25C_60RH (long-term) / 30C_65RH (intermediate) / 40C_75RH (accelerated)
- `temperatuur_C`, `rel_vochtigheid_pct` - opslagcondities
- `tijdstip_maanden` - meettijdstip (0, 1, 2, 3, 6, 9, 12, 18, 24, 36 maanden)
- `gehalte_pct` - **target (regressie):** gehalte actief ingrediënt (%), spec: > 95
- `onzuiverheid_A_pct`, `onzuiverheid_B_pct` - individuele onzuiverheden (%)
- `totaal_onzuiverheden_pct` - totaal onzuiverheden (%), spec: < 2.0
- `watergehalte_pct` - watergehalte (%)
- `dissolutie_pct` - dissolutie (%), spec: > 75 (alleen tabletten/capsules)
- `uiterlijk_score` - visuele score (1-5, 5 = perfect)
- `out_of_spec` - **target (classificatie):** 1 = buiten specificatie

**ML-toepassingen:** Degradatiekinetiek modellering, shelf life voorspelling, Arrhenius-modellen, vergelijking formuleringen.

**Bijzonderheden:** Suspensies degraderen sneller (form_factor=1.8). Versnelde condities (40°C/75%RH) tonen snellere afbraak.

---

### 12. Warmtewisselaar Fouling (`warmtewisselaar_fouling.csv`)

**Context:** Continue monitoring van een shell-and-tube warmtewisselaar in een chemische plant. Uurlijkse data over 1 jaar, met periodieke schoonmaakacties.

**Kolommen:**
- `datum` - datum en tijd
- `dag_sinds_reiniging` - dagen sinds laatste schoonmaakactie
- `hot_inlet_C`, `hot_outlet_C` - hete zijde temperaturen (°C)
- `hot_flow_m3h` - hete zijde debiet (m³/h)
- `cold_inlet_C`, `cold_outlet_C` - koude zijde temperaturen (°C)
- `cold_flow_m3h` - koude zijde debiet (m³/h)
- `drukval_hot_bar`, `drukval_cold_bar` - drukvallen (bar), stijgen met fouling
- `U_actueel_Wm2K` - actuele warmtedoorgangscoëfficiënt (W/m²·K)
- `effectiviteit` - thermische effectiviteit warmtewisselaar
- `fouling_factor` - **target (regressie):** fouling weerstand
- `onderhoudsstatus` - **target (classificatie):** ok / waarschuwing / gepland

**ML-toepassingen:** Predictive maintenance, regressie (fouling voorspellen), tijdreeksanalyse, optimale onderhoudsplanning.

**Bijzonderheden:** Niet-lineaire fouling groei (~t^1.3). Seizoenseffecten in inlet-temperaturen. Periodieke schoonmaak elke ~90 dagen.

---

### 13. Continue Polymerisatie (`continue_polymerisatie.csv`)

**Context:** Continue polyethyleenreactor met grade-transities. Per-minuut data over 14 dagen. PID-geregelde temperatuur en druk, 4 productgrades, procesverstoringen.

**Kolommen:**
- `timestamp` - tijdstip (per minuut)
- `grade` - huidige productgrade: HDPE_A / HDPE_B / LDPE_C / LLDPE_D
- `in_transitie` - 1 = grade-transitie bezig (2 uur per transitie)
- `temp_setpoint_C`, `temp_actueel_C` - temperatuur setpoint en actueel (°C)
- `temp_sensor_C` - ruwe sensorwaarde (bevat sensor spike op min 15000)
- `druk_setpoint_bar`, `druk_actueel_bar` - druk setpoint en actueel (bar)
- `katalysator_flow_kgh` - katalysatordosering (kg/h)
- `monomeer_flow_kgh` - monomeerdebiet (kg/h)
- `waterstof_flow_kgh` - waterstofdebiet voor MFI-controle (kg/h)
- `comonomeer_flow_kgh` - comonomeerdebiet (hoog bij LLDPE)
- `koelwater_inlet_C`, `koelwater_outlet_C`, `koelwater_flow_m3h` - koelwatersysteem
- `melt_flow_index_g10min` - **target (soft sensor):** MFI (g/10min)
- `viscositeit_Pas` - **target (soft sensor):** smeltviscositeit (Pa·s)
- `dichtheid_gcm3` - productdichtheid (g/cm³)
- `vermogen_kW` - energieverbruik (kW)
- `verstoring` - label: normaal / katalysator_puls / koelwater_uitval / monomer_verontreiniging / sensor_fout

**ML-toepassingen:** Soft sensor ontwikkeling (MFI/viscositeit), grade-transitie optimalisatie, anomalie-detectie, PID-analyse, sensor fault detection.

**Bijzonderheden:** 4 grade-transities met smooth setpoint changes. 4 geinjecteerde procesverstoringen met cascade-effecten. Sensor spike (fout vs. echte verstoring).

---

### 14. Continue Droging (`continue_droging.csv`)

**Context:** Wervelbeddroger in farmaceutische productie. 30 opeenvolgende batches (~5 uur elk), data elke 30 seconden. Inline NIR-metingen.

**Kolommen:**
- `timestamp` - tijdstip (elke 30 sec)
- `batch_id` - batch identifier
- `tijd_sec` - tijd binnen de batch (sec)
- `inlet_lucht_temp_C`, `inlet_lucht_temp_sp_C` - inlaatluchttemperatuur actueel en setpoint
- `outlet_lucht_temp_C` - uitlaatluchttemperatuur (stijgt naarmate product droger wordt)
- `product_temp_C` - producttemperatuur
- `outlet_luchtvochtigheid_pct` - vochtigheid uitlaatlucht (%)
- `drukval_mbar` - drukval over het wervelbed (mbar)
- `luchtdebiet_m3h` - luchtdebiet (m³/h)
- `bedmassa_kg` - massa product in droger (kg)
- `vochtgehalte_pct` - **target (soft sensor):** werkelijk vochtgehalte (%)
- `NIR_vochtgehalte_pct` - inline NIR-meting (elke 5 min, anders leeg)
- `droogsnelheid_pctmin` - momentane droogsnelheid (%/min)
- `energie_kW` - energieverbruik (kW)
- `fase` - **target (classificatie):** opwarmen / constant_rate / falling_rate / eindpunt

**ML-toepassingen:** Soft sensor (vochtgehalte uit procesdata), eindpuntdetectie, droogcurve modellering, vergelijking droogcondities, energieoptimalisatie.

**Bijzonderheden:** Exponentieel droogprofiel. Variatie tussen batches (temperatuur, luchtstroom, initieel vochtgehalte). NIR als referentiemeting met grotere meetinterval.

---

### 15. Continue Menging (`continue_menging.csv`)

**Context:** Continue farmaceutische poedermenger met 3 componenten (API, vulstof, glijmiddel). Loss-in-weight feeders, inline NIR-spectroscopie. Data elke 10 seconden, 48 uur.

**Kolommen:**
- `timestamp` - tijdstip (elke 10 sec)
- `API_voeding_kgh` - API feeder debiet (kg/h), target: 5.0
- `excipient_voeding_kgh` - vulstof feeder debiet (kg/h), target: 45.0
- `glijmiddel_voeding_kgh` - glijmiddel feeder debiet (kg/h), target: 0.5
- `totaal_debiet_kgh` - totaal massadebiet (kg/h)
- `menger_snelheid_RPM` - mengsnelheid (RPM)
- `menger_torque_Nm` - torque op de menger (Nm)
- `menger_temp_C` - temperatuur in menger (°C)
- `verblijftijd_sec` - verblijftijd in menger (sec)
- `API_fractie` - berekende API-fractie in het mengsel
- `RSD_pct` - **target (soft sensor):** relatieve standaarddeviatie menguniformiteit (%)
- `NIR_RSD_pct` - inline NIR-gemeten RSD (%)
- `spec_conform` - **target (classificatie):** 1 = RSD < 5% (binnen spec)
- `verstoring` - label: normaal / feeder_blokkade / segregatie / hoog_vochtgehalte

**ML-toepassingen:** Menguniformiteit voorspelling, feeder fault detection, real-time release testing (RTRT), procescontrole, multivariate monitoring.

**Bijzonderheden:** Feeder refill-events (elke ~2 uur). 3 verschillende procesverstoringen. Feederblokkade toont cascade-effect op uniformiteit.

---

### 16. Geintegreerde Productielijn (`geintegreerde_productielijn.csv`)

**Context:** Complete chemische productielijn: reactor -> flash separator -> zuiveringskolom. Data elke 5 minuten, 60 dagen. Cascade-effecten tussen units.

**Kolommen (prefix R=Reactor, S=Separator, K=Kolom):**
- `timestamp` - tijdstip (elke 5 min)
- `R_voeding_flow_kgh`, `R_voeding_temp_C`, `R_voeding_zuiverheid` - voedingscondities
- `R_temp_sp_C`, `R_temp_actueel_C` - reactor temperatuur (setpoint volgt weekcyclus)
- `R_druk_bar` - reactordruk (bar)
- `R_niveau_pct` - reactorniveau (%)
- `R_conversie` - conversiegraad reactor
- `S_temp_C`, `S_druk_bar`, `S_niveau_pct` - separator condities
- `S_damp_flow_kgh`, `S_vloeistof_flow_kgh` - scheiding damp/vloeistof
- `S_vloeistof_zuiverheid` - zuiverheid vloeibare stroom
- `K_reflux_ratio`, `K_reboiler_duty_kW` - kolominstellingen
- `K_top_temp_C`, `K_bodem_temp_C`, `K_drukval_bar` - kolomtemperaturen en drukval
- `product_zuiverheid` - **target (regressie):** eindzuiverheid product
- `product_flow_kgh` - productdebiet (kg/h)
- `totaal_energie_kW` - totaal energieverbruik (kW)
- `on_spec` - **target (classificatie):** 1 = zuiverheid > 99.0%
- `event` - label: normaal / voedingsverontreiniging / koelwater_storing / kolom_flooding / katalysator_deactivatie

**ML-toepassingen:** Multivariate procesmonitoring (PCA/PLS), root cause analysis, cascade fault detection, soft sensor, energieoptimalisatie.

**Bijzonderheden:** 4 procesverstoringen met cascade-effecten tussen units (vertraging zichtbaar). Voedingsverontreiniging propageert door reactor -> separator -> kolom. Geleidelijke katalysator deactivatie over 8 dagen.

---

### 17. Compressor Monitoring (`compressor_monitoring.csv`)

**Context:** Centrifugaalcompressor in chemische plant. Trillings-, temperatuur- en procesdata. Elke minuut, 90 dagen. Geleidelijke lagerdegradatie.

**Kolommen:**
- `timestamp` - tijdstip (elke minuut)
- `toerental_RPM` - draaisnelheid (~12000 RPM)
- `aanzuigdruk_bar`, `persdruk_bar` - aanzuig- en persdruk (bar)
- `drukverhouding` - persdruk / aanzuigdruk
- `aanzuigtemp_C`, `perstemp_C` - temperaturen (°C)
- `debiet_m3h` - gasdebiet (m³/h)
- `lager_DE_temp_C`, `lager_NDE_temp_C` - lagertemperaturen drive-end en non-drive-end (°C)
- `trillingen_DE_x_mms`, `trillingen_DE_y_mms` - trillingen drive-end (mm/s RMS)
- `trillingen_NDE_x_mms`, `trillingen_NDE_y_mms` - trillingen non-drive-end (mm/s RMS)
- `olie_druk_bar`, `olie_temp_C` - smeerolie parameters
- `olie_kwaliteit_pct` - oliekwaliteit (%, daalt geleidelijk)
- `isentropisch_rendement` - **target (regressie):** compressor efficiency
- `vermogen_kW` - opgenomen vermogen (kW)
- `machine_status` - **target (classificatie):** normaal / surge / lichte_degradatie / waarschuwing / alarm

**ML-toepassingen:** Predictive maintenance, remaining useful life (RUL), trillingsanalyse, surge-detectie, degradatie-modellering, multivariate monitoring.

**Bijzonderheden:** Geleidelijke lagerdegradatie vanaf dag 50 (exponentieel toenemende trillingen). Surge-event op dag 35 (10 min). Oliekwaliteit daalt lineair. 129.600 datapunten.

---

### 18. CHO Celcultuur Perfusie (`cho_celcultuur_perfusie.csv`)

**Context:** Perfusie-bioreactor met CHO-cellen (Chinese Hamster Ovary) voor productie van monoklonale antilichamen (mAb). Uurlijkse data over 60 dagen met celretentiefilter.

**Kolommen:**
- `timestamp`, `dag` - tijdstip en procesdagen
- `temp_sp_C`, `temp_actueel_C` - temperatuur (37°C -> 33°C shift op dag 20)
- `pH_sp`, `pH_actueel` - pH setpoint en actueel
- `DO_sp_pct`, `DO_actueel_pct` - opgeloste zuurstof (% luchtverzadiging)
- `viable_celdichtheid_celmL` - **target (regressie):** levende celdichtheid (cellen/mL)
- `totale_celdichtheid_celmL` - totale celdichtheid
- `viabiliteit_pct` - **target (regressie):** percentage levende cellen
- `glucose_mmolL`, `glutamine_mmolL` - voedingsstoffen
- `lactaat_mmolL`, `ammonium_mmolL` - metabolieten (remmend bij hoge concentratie)
- `osmolaliteit_mOsmkg` - osmolaliteit (mOsm/kg)
- `titer_gL` - **target (regressie):** mAb productconcentratie (g/L)
- `perfusiesnelheid_VVD` - perfusiesnelheid (volumes per dag)
- `bleed_rate_VVD` - bleed rate voor celdichtheidscontrole
- `lucht_sparge_Lmin`, `O2_sparge_Lmin`, `CO2_sparge_Lmin` - gasdoseringen
- `turbiditeit_AU` - inline turbiditeit (correleert met celdichtheid)
- `filter_druk_bar` - celretentiefilter druk (stijgt met fouling)
- `cultuur_fase` - lag / groei / stationair / productie / afsterving
- `event` - normaal / pH_excursie / DO_dip / filter_fouling / filter_kritiek

**ML-toepassingen:** Titervoorspelling, viabiliteitsmonitoring, metabolietmodellering, cultuurverloop-classificatie, filter fouling voorspelling.

**Bijzonderheden:** Temperatuurshift op dag 20 (stimuleert productie). Logistische celgroei met metabolietremming. Geleidelijke filter fouling vanaf dag 45.

---

### 19. Chromatografische Zuivering (`chromatografie_zuivering.csv`)

**Context:** Downstream processing van mAb via chromatografie. 200 runs over Protein A, kationenwisseling (CEX) en anionenwisseling (AEX) kolommen.

**Kolommen:**
- `run_id` - run identifier
- `kolom_type` - ProteinA / CEX / AEX
- `kolom_leeftijd_cycli` - aantal cycli op de hars (1-300)
- `bedhoogte_cm`, `kolom_diameter_cm` - kolomdimensies
- `feed_titer_gL` - feed concentratie (g/L)
- `feed_volume_CV` - belading in kolomvolumes
- `feed_HCP_ppm`, `feed_DNA_ppm`, `feed_aggregaat_pct` - feed onzuiverheden
- `load_snelheid_cmh`, `load_pH`, `load_conductiviteit_mScm` - beladingscondities
- `was_volume_CV` - wasvolume
- `elutie_pH`, `elutie_conductiviteit_mScm`, `elutie_snelheid_cmh` - elutiecondities
- `temperatuur_C` - procestemperatuur (4/15/22°C)
- `DBC_gL` - dynamische bindingscapaciteit (g/L hars)
- `doorbraak_pct` - doorbraakpercentage
- `drukval_bar`, `HETP_cm` - kolomperformance
- `UV_piekhoogte_mAU`, `UV_piekbreedte_CV`, `UV_asymmetrie` - UV280 piekkarakteristieken
- `pool_volume_mL`, `pool_concentratie_gL` - eluaat pool
- `pool_HCP_ppm`, `pool_aggregaat_pct` - **targets (regressie):** pool kwaliteit
- `stap_opbrengst_pct` - **target (regressie):** opbrengst per stap

**ML-toepassingen:** Opbrengst- en zuiverheidsvoorspelling, harslevensduur modellering, procesoptimalisatie, vergelijking kolomtypes.

**Bijzonderheden:** Bindingscapaciteit daalt met kolomleeftijd. Drie kolomtypes met verschillende scheidingskarakteristieken. Rijke feature set (30 kolommen).

---

### 20. Lyofilisatie Cyclus (`lyofilisatie_cyclus.csv`)

**Context:** Vriesdroogproces voor biofarmaceutische producten. 60 batches met volledige cyclus (invriezen -> primair drogen -> secundair drogen). Per-minuut data, ~40 uur per batch.

**Kolommen:**
- `batch_id` - batch identifier
- `tijd_min` - tijd in de cyclus (min)
- `fase` - **target (classificatie):** invriezen / hold_bevroren / annealing / primair_drogen / opwarmen_secundair / secundair_drogen
- `plaat_temp_C` - plaattemperatuur (°C)
- `product_temp_C` - producttemperatuur (°C, laag door sublimatie)
- `kamer_druk_mTorr` - kamerdruk (mTorr, alleen tijdens drogen)
- `condenser_temp_C` - condensertemperatuur (°C)
- `pirani_ratio` - **target (eindpuntdetectie):** Pirani/capacitance drukratio (daalt naar 1.0 bij droog)
- `formulering` - sucrose / trehalose / mannitol
- `vaste_stof_pct` - vaste stofgehalte (% w/v)
- `vulvolume_mL` - vulvolume per vial (mL)
- `eiwit_conc_mgmL` - eiwitconcentratie (mg/mL)
- `invriessnelheid_Cmin` - invriessnelheid (°C/min)

**ML-toepassingen:** Eindpuntdetectie primair drogen, faseherkenning, cyclusoptimalisatie, procesvoorspelling op basis van formulering.

**Bijzonderheden:** Pirani ratio als eindpuntindicator (1.5 -> 1.0). Variatie in formulering, vulvolume en invriessnelheid. 118.672 datapunten.

---

### 21. PID Regelkringen (`pid_regelkringen.csv`)

**Context:** 20 PID-regelkringen in een chemische plant met verschillende tuning-kwaliteiten. Setpoint, proceswaarde en regeluitgang elke 5 seconden, 4 uur per loop.

**Kolommen:**
- `loop_tag` - tag van de regelkring (bijv. TIC-101, FIC-201)
- `loop_type` - temperatuur / debiet / druk / niveau / pH / concentratie
- `tuning_kwaliteit` - **target (classificatie):** goed / oscillerend / traag / agressief / sticking_valve / quantized / saturatie / niet_lineair / dood_tijd / sensor_ruis
- `tijd_sec` - tijd (sec)
- `setpoint` - setpoint waarde
- `proceswaarde` - actuele proceswaarde (PV)
- `regeluitgang_pct` - regelaaruitgang (MV, 0-100%)
- `afwijking` - SP - PV
- `abs_afwijking` - |SP - PV|

**ML-toepassingen:** Control loop performance monitoring, oscillatie-detectie, sticking valve detectie, tuning-classificatie, feature engineering (IAE, ISE, overshoot).

**Bijzonderheden:** 10 verschillende tuning-problemen: oscillatie (te hoge gain), traag (te lage gain), sticking valve (deadband), quantized output, saturatie, niet-lineaire klep, dode tijd, sensor ruis. Setpoint stappen om respons te testen.

---

### 22. Alarmbeheer Logboek (`alarmbeheer_logboek.csv`)

**Context:** Alarmlogboek van een chemische plant over 6 maanden. 20 alarmtags met verschillende prioriteiten en frequenties. Inclusief alarm floods en nuisance alarmen.

**Kolommen:**
- `alarm_id` - uniek alarm ID
- `timestamp` - tijdstip van het alarm
- `tag` - alarmpunt (bijv. TAH-101, PAH-201)
- `prioriteit` - kritiek / hoog / medium / laag
- `type` - temperatuur / druk / debiet / niveau / equipment / analyse / veiligheid / communicatie
- `status` - actief / acknowledged / teruggekeerd
- `duur_sec` - alarmduur (sec)
- `is_nuisance` - **target (classificatie):** 1 = nuisance alarm (chattering/frequent)
- `in_alarm_flood` - 1 = alarm tijdens flood event
- `operator_actie` - acknowledge / suppress / actie_genomen / geen
- `dag_type` - werkdag / weekend
- `shift` - dag / avond / nacht

**ML-toepassingen:** Alarm flood detectie/voorspelling, nuisance alarm identificatie, alarm rationalisatie, patroonanalyse, shift-analyse.

**Bijzonderheden:** ~5% van de dagen bevat alarm flood events. 4 nuisance alarmtags (chattering, te frequente alarmen). Verschil werkdag/weekend. ISA-18.2 alarmmanagement context.

---

### 23. Energie & Utiliteiten (`energie_utiliteiten.csv`)

**Context:** Stoom, elektriciteit, koelwater, perslucht en stikstofverbruik van een chemische plant. Per 15 minuten, 1 jaar. Seizoensgebonden patronen.

**Kolommen:**
- `timestamp` - tijdstip (elke 15 min)
- `buitentemp_C` - omgevingstemperatuur (°C, seizoensgebonden)
- `productiebelasting` - relatieve productiebelasting (0.3-1.0)
- `stoom_tonh` - stoomverbruik (ton/h)
- `stoom_druk_bar` - stoomdruk (bar)
- `boiler_rendement` - ketelrendement (0.78-0.92)
- `gasverbruik_Nm3h` - aardgasverbruik (Nm³/h)
- `elektriciteit_kW` - **target (regressie):** elektriciteitsverbruik (kW)
- `piekbelasting` - **target (classificatie):** 1 = boven piekgrens
- `koelwater_m3h` - koelwaterverbruik (m³/h)
- `koelwater_aanvoer_C`, `koelwater_retour_C` - koelwatertemperaturen
- `koeltoren_approach_C` - koeltoren approach temperatuur
- `perslucht_Nm3h` - persluchtverbruik (Nm³/h)
- `compressor_vermogen_kW` - compressorvermogen (kW)
- `stikstof_Nm3h` - stikstofverbruik (Nm³/h)
- `energiekost_EURh` - **target (regressie):** totale energiekosten (EUR/h)
- `CO2_kgh` - CO₂-uitstoot (kg/h)
- `onderhoudsstop` - 1 = tijdens onderhoudsstop (4x per jaar)

**ML-toepassingen:** Energievoorspelling, piekdetectie, kostenoptimalisatie, seizoensanalyse, CO₂-reductie, anomalie-detectie, onderhoudsstop impact.

**Bijzonderheden:** Dag/nacht en werkdag/weekend patronen. Seizoenseffect op koeling (zomer) en verwarming (winter). 4 onderhoudsstops per jaar met gereduceerde belasting. 35.040 datapunten over heel 2025.

---

### 24. Machine Vision Inspectie (`machine_vision_inspectie.csv`)

**Context:** Geëxtraheerde beeldfeatures van visuele kwaliteitsinspectie van vials, tabletten en capsules op 3 productielijnen. Gesimuleerde CNN/beeldverwerkingsoutput.

**Kolommen:**
- `inspectie_id` - uniek inspectie-ID
- `product_type` - vial / tablet / capsule
- `productielijn` - Lijn_1 / Lijn_2 / Lijn_3
- `lijnsnelheid_per_min` - productiesnelheid
- `oppervlakte_px`, `omtrek_px` - geometrische features
- `circulariteit`, `aspectverhouding`, `compactheid` - vormfeatures
- `gem_intensiteit`, `std_intensiteit`, `kleuruniformiteit` - kleurfeatures
- `contrast`, `correlatie`, `energie`, `homogeniteit` - textuurfeatures (GLCM)
- `randdichtheid`, `ruwheid` - oppervlaktefeatures
- `aantal_contouren` - aantal gedetecteerde contouren
- `confidence_score` - modelvertrouwen (0-1)
- `defect_type` - **target (multiclass):** geen / kras / verkleuring / barst / deeltje / vormafwijking / vulniveau
- `goedgekeurd` - **target (binair):** 1 = goedgekeurd (~85%)

**ML-toepassingen:** Defectclassificatie, anomalie-detectie, confidence kalibratie, feature importance, ongebalanceerde classificatie.

**Bijzonderheden:** 6 defecttypes met elk eigen feature-signature. ~15% defect rate. Vulniveau-defect alleen bij vials. Confidence score weerspiegelt modelzekerheid.

---

### 25. NIR Spectroscopie (`nir_spectroscopie.csv`)

**Context:** NIR-spectra (256 golflengten, 900-2500 nm) van farmaceutische poeders, granulaten en tabletten voor inline kwaliteitsbepaling.

**Kolommen:**
- `sample_id`, `batch_id` - identificatie
- `sample_type` - poeder / granulaat / tablet
- `API_pct` - **target (regressie):** API-concentratie (5-50% w/w)
- `vochtgehalte_pct` - **target (regressie):** vochtgehalte (0.5-8%)
- `deeltjesgrootte_um` - deeltjesgrootte (20-200 µm, beinvloedt scattering)
- `wl_900nm` t/m `wl_2500nm` - **input (256 features):** absorbantie bij elke golflengte

**ML-toepassingen:** PLS/PCR multivariate kalibratie, wavelength selectie (variable importance), scatteringcorrectie (SNV/MSC), transfer learning tussen sample types.

**Bijzonderheden:** Realistische absorptiebanden: API (1200, 1680, 2200 nm), water (1450, 1940 nm). Deeltjesgrootte veroorzaakt baseline shift (scattering). 262 kolommen - ideaal voor dimensiereductie.

---

### 26. Klep Diagnostiek (`klep_diagnostiek.csv`)

**Context:** Smart valve diagnostiek van 50 regelkleppen (globe/butterfly/ball) met HART/fieldbus parameters. 20 tests per klep over ~10 jaar.

**Kolommen:**
- `klep_id` - klep identifier
- `test_nr`, `leeftijd_maanden` - testnummer en leeftijd
- `klep_type` - globe / butterfly / ball
- `diameter_inch` - klepdiameter (2/4/6/8 inch)
- `actuator_type` - pneumatisch / elektrisch
- `cycli_totaal` - totaal aantal slagen
- `procestemperatuur_C` - procestemperatuur
- `pakkingwrijving_N` - stangafdichtingswrijving (N)
- `slagafwijking_pct` - afwijking van nominale slag (%)
- `staptijd_sec` - tijd voor volledige slag (sec)
- `dodeband_pct`, `hysterese_pct` - regelnauwkeurigheid (%)
- `luchtverbruik_Lmin` - luchtverbruik (alleen pneumatisch)
- `zittinglek_Lmin` - lekkage bij gesloten klep
- `overshoot_pct`, `undershoot_pct`, `settling_time_sec` - staprespons
- `voedingsdruk_bar` - actuator voedingsdruk
- `dp_ratio` - drukvalverhouding over klep
- `faalmodus` - **target (classificatie):** gezond / pakkinglek / sticking / erosie / actuator_zwak / positioner_drift / cavitatie

**ML-toepassingen:** Faalmodusclassificatie, degradatie-modellering, remaining useful life, feature engineering over tijd.

**Bijzonderheden:** 7 faalmodi met elk eigen degradatiepatroon. Progressieve verslechtering zichtbaar over opeenvolgende tests. 35% gezonde kleppen als referentie.

---

### 27. CIP Reiniging (`cip_reiniging.csv`)

**Context:** Clean-in-Place reinigingscycli in farmaceutische/food productie. 100 cycli met 6 fases (voorspoeling -> loog -> tussenspoeling -> zuur -> naspoeling -> eindspoeling). Data elke 10 seconden.

**Kolommen:**
- `cyclus_id` - cyclus identifier
- `equipment` - reactor_500L / reactor_2000L / tank_5000L / leiding_DN50
- `vorig_product` - product dat werd geproduceerd voor reiniging
- `vervuilingsgraad` - relatieve vervuiling (1-5)
- `fase` - **target (classificatie):** voorspoeling / loogfase / tussenspoeling / zuurfase / naspoeling / eindspoeling
- `tijd_sec`, `tijd_in_fase_sec` - totale en fase-relatieve tijd
- `temperatuur_C` - reinigingstemperatuur
- `conductiviteit_mScm` - **target (eindpunt):** conductiviteit (mS/cm, moet < 1.5 bij eindspoel)
- `turbiditeit_NTU` - troebelheid (NTU)
- `pH` - zuurgraad
- `TOC_mgL` - **target (eindpunt):** Total Organic Carbon (mg/L, moet < 5 bij einde)
- `debiet_m3h` - spoeldebiet
- `druk_bar` - leidingdruk
- `NaOH_pct`, `HNO3_pct` - chemicaliënconcentraties
- `cyclus_resultaat` - **target (classificatie):** geslaagd / gefaald

**ML-toepassingen:** Reinigingseindpunt voorspelling, cyclustijdoptimalisatie, vergelijking equipment/producten, water- en chemicaliënbesparing.

**Bijzonderheden:** Exponentieel uitspoel-profiel. Vervuilingsgraad beïnvloedt reinigingstijd. Variatie per equipment type en vorig product.

---

### 28. Cleanroom Monitoring (`cleanroom_monitoring.csv`)

**Context:** Environmental monitoring van 6 farmaceutische cleanrooms (GMP Grade A/B/C/D). Deeltjestelling, klimaat en druk. Elke 30 minuten per ruimte, 90 dagen.

**Kolommen:**
- `timestamp` - tijdstip
- `ruimte` - Room_A1, A2, B1, B2, C1, D1
- `GMP_grade` - A / B / C / D
- `deeltjes_05um_per_m3` - deeltjes >= 0.5 µm per m³
- `deeltjes_50um_per_m3` - deeltjes >= 5.0 µm per m³
- `limiet_05um`, `limiet_50um` - GMP limieten per grade
- `temperatuur_C` - ruimtetemperatuur (target: 20 ± 2°C)
- `rel_vochtigheid_pct` - relatieve vochtigheid (target: 30-65%)
- `drukverschil_kPa` - overdruk t.o.v. omgeving
- `min_drukverschil_kPa` - minimaal vereist drukverschil
- `in_specificatie` - **target (classificatie):** 1 = alle parameters binnen spec
- `alarm` - **target (multiclass):** geen / deeltjes_alarm / druk_alarm / temp_alarm / rv_alarm
- `event` - normaal / deuropening / HVAC_storing / schoonmaak

**ML-toepassingen:** Anomalie-detectie, alarmvoorspelling, dag/nacht patroonanalyse, vergelijking tussen grades, HVAC-optimalisatie.

**Bijzonderheden:** Strenge limieten voor Grade A (3520 deeltjes/m³) vs. Grade D (3.520.000). Dag/nacht personeelseffect. HVAC-storingen en deuropeningen als events.

---

### 29. Golden Batch Coating (`golden_batch_coating.csv`)

**Context:** 80 batches van een farmaceutisch filmcoatingproces. Vergelijking met het ideale "golden batch" temperatuur- en sprayprofiel. Per minuut, ~2 uur per batch.

**Kolommen:**
- `batch_id` - batch identifier
- `tijd_min` - tijd in de batch (min)
- `inlet_temp_C` - inlaatluchttemperatuur (°C)
- `exhaust_temp_C` - uitlaatluchttemperatuur (°C)
- `bed_temp_C` - productbedtemperatuur (°C)
- `spray_rate_gmin` - spraysnelheid (g/min)
- `pan_snelheid_RPM` - pansnelheid (RPM)
- `exhaust_humidity_pct` - uitlaatvochtigheid (%)
- `drukval_mbar` - drukval over trommel (mbar)
- `gewichtstoename_pct` - cumulatieve gewichtstoename coating (%)
- `kern_gewicht_mg` - tabletgewicht voor coating (mg)
- `coating_conc_pct` - coating oplossingsconcentratie (%)
- `afwijking_temp` - afstand tot golden batch temperatuur
- `afwijking_spray` - afstand tot golden batch sprayrate
- `batch_kwaliteit` - **target (classificatie):** goed (65%) / grensgeval (20%) / afwijkend (15%)

**ML-toepassingen:** Multivariate batch analyse (MPCA/MSPC), golden batch vergelijking, real-time batchkwaliteit voorspelling, early fault detection.

**Bijzonderheden:** Goed/grensgeval/afwijkend batches met verschillende offset en timing variatie. Opwarmprofiel, spray opbouw en steady-state fases.

---

### 30. MPC Destillatie (`mpc_destillatie.csv`)

**Context:** Model Predictive Control data van een destillatiekolom. 4 gecontroleerde variabelen (CV), 3 gemanipuleerde variabelen (MV), 2 storingsvariabelen (DV). Per minuut, 48 uur.

**Kolommen:**
- `timestamp` - tijdstip
- `CV_top_zuiverheid`, `CV_bodem_zuiverheid` - productzuiverheden
- `CV_kolom_druk_bar` - kolomdruk
- `CV_condenser_niveau_pct` - condenserniveau
- `SP_*` - setpoints voor elke CV
- `MV_reflux_flow_kgh` - refluxdebiet (MV 1)
- `MV_reboiler_duty_kW` - reboilervermogen (MV 2)
- `MV_feed_flow_kgh` - voedingsdebiet (MV 3)
- `DV_feed_compositie` - voedingssamenstelling (verstoring)
- `DV_omgevingstemp_C` - omgevingstemperatuur (verstoring)
- `mpc_status` - **target (classificatie):** actief / operator_override / infeasible
- `sp_verandering` - 1 = setpoint wijziging op dit moment
- `cv_constraint_overtreding` - 1 = CV buiten constraints
- `mv_constraint_actief` - 1 = MV op limiet
- `objectieffunctie` - MPC kostenfunctie waarde

**ML-toepassingen:** MPC performance analyse, constraint voorspelling, vergelijking MPC vs PID, storingsinvloed analyse, objectieffunctie modellering.

**Bijzonderheden:** 3 setpoint veranderingen, operator override periode (30 min), MPC infeasibility event. CV en MV constraints zichtbaar in data.

---

### 31. Digital Twin Validatie (`digital_twin_validatie.csv`)

**Context:** Vergelijking tussen een fysiek procesmodel (digital twin) en werkelijke reactordata. Het model veroudert geleidelijk doordat het geen rekening houdt met katalysatordegradatie. Elke 10 minuten, 30 dagen.

**Kolommen:**
- `timestamp` - tijdstip
- `voeding_flow_kgh`, `voeding_temp_C` - procesomstandigheden
- `reactor_temp_werkelijk_C`, `conversie_werkelijk`, `product_temp_werkelijk_C`, `energie_werkelijk_kW` - werkelijke proceswaarden
- `reactor_temp_model_C`, `conversie_model`, `product_temp_model_C`, `energie_model_kW` - digital twin voorspellingen
- `residu_temp_C`, `residu_conversie`, `residu_energie_kW` - model-werkelijk residuen
- `model_confidence` - vertrouwensscore van het model (0-1)
- `drift_score` - **target (regressie):** mate van model-werkelijkheid afwijking
- `drift_gedetecteerd` - **target (classificatie):** 1 = significante drift
- `event` - normaal / katalysator_shift (dag 20)

**ML-toepassingen:** Concept drift detectie, modelkalibratie, residuanalyse, adaptief modelleren, anomalie-detectie in model-werkelijkheid verschil.

**Bijzonderheden:** Geleidelijke modelveroudering door katalysatordegradatie. Abrupte procesverandering op dag 20. Model heeft licht andere coëfficiënten dan werkelijkheid.

---

### 32. Operator Logboeken (`operator_logboeken.csv`)

**Context:** Shift logboeken van operators in een chemische plant over 1 jaar. Elke shift (dag/avond/nacht) bevat gestructureerde tekst met notities, incidenten en handover-informatie.

**Kolommen:**
- `log_id` - uniek logboek ID
- `datum` - datum
- `shift` - dag / avond / nacht
- `operator` - operatornaam
- `dag_type` - werkdag / weekend
- `logboek_tekst` - **input (NLP):** vrije tekst met meerdere entries gescheiden door "|"
- `handover_notitie` - overdrachtsnotitie voor volgende shift
- `n_entries` - aantal logboekentries
- `n_alarmen` - aantal alarmen tijdens shift
- `productie_ton` - productie (ton)
- `OEE` - Overall Equipment Effectiveness
- `heeft_incident` - **target (classificatie):** 1 = incident tijdens shift
- `ernst` - normaal / laag / medium / hoog / kritiek
- `categorie` - routine / storing / kwaliteit / veiligheid / proces

**ML-toepassingen:** Tekstclassificatie (incident detectie), NER (equipment, parameters), sentimentanalyse, trend-detectie over tijd, shift-vergelijking.

**Bijzonderheden:** Mix van normale en incident-gerelateerde logboeken. Variatie in schrijfstijl per operator. Weekend/nacht shifts hebben minder activiteit.

---

### 33. Multi-product Receptoptimalisatie (`recept_optimalisatie.csv`)

**Context:** 5 farmaceutische producten op 1 productielijn. 800 batches met variërende grondstofkwaliteit, receptinstellingen en seizoenseffecten.

**Kolommen:**
- `batch_id`, `datum`, `seizoen` - identificatie en timing
- `product` - Paracetamol_500mg / Ibuprofen_400mg / Aspirine_300mg / Metformin_850mg / Omeprazol_20mg
- `leverancier` - Sup_A / Sup_B / Sup_C
- `API_zuiverheid_pct`, `API_vochtgehalte_pct`, `API_deeltjesgrootte_d50_um`, `API_bulkdichtheid_gmL` - grondstofeigenschappen
- `granulatie_water_pct`, `mengtijd_min`, `perskracht_kN`, `perssnelheid_RPM` - receptparameters
- `droogtemp_C`, `droogtijd_min`, `coatingtijd_min` - procesparameters
- `ruimte_temp_C`, `ruimte_RV_pct` - omgevingscondities
- `gewicht_mg`, `hardheid_N`, `dissolutie_pct`, `brosheid_pct` - productkwaliteit
- `gehalte_uniformiteit_RSD`, `restvochtgehalte_pct` - aanvullende kwaliteit
- `coating_gewichtstoename_pct`, `uiterlijk_score` - coating resultaat
- `cyclustijd_min`, `batchkosten_EUR` - efficiëntie
- `goedgekeurd` - **target (classificatie):** 1 = voldoet aan alle specs

**ML-toepassingen:** Product-specifieke modellering, multi-task learning, receptoptimalisatie, seizoenscorrectie, leverancierseffect analyse, kostenoptimalisatie.

**Bijzonderheden:** 5 producten met verschillende complexiteit. Seizoenseffect op vochtigheid. Interactie grondstofkwaliteit x receptinstellingen. 29 features.

---

### 34. Reinforcement Learning Reactor (`rl_reactor_control.csv`)

**Context:** Offline RL dataset voor reactortemperatuurregeling. 200 episodes met data van 4 verschillende control policies (PID conservatief/agressief, expert, random).

**Kolommen:**
- `episode`, `stap` - episode en stap index
- `policy` - PID_conservatief / PID_agressief / expert / random
- `setpoint_C` - temperatuur setpoint (145/150/155/160°C)
- `state_temp_C`, `state_conc_molL`, `state_coolant_C` - toestandsvariabelen
- `state_error_C` - tracking error
- `state_verstoring` - externe verstoring
- `action_coolant_adj` - **actie:** koelwateraanpassing (-10 tot +10 L/min)
- `reward_tracking`, `reward_energie`, `reward_constraint` - reward componenten
- `reward_totaal` - **target:** totale beloning per stap
- `cumulatief_reward` - cumulatieve beloning
- `next_temp_C`, `next_conc_molL`, `next_coolant_C` - volgende toestand
- `done` - 1 = episode afgelopen (terminal state of constraint violated)

**ML-toepassingen:** Offline RL (batch RL), policy evaluation, imitation learning, vergelijking control strategieën, reward shaping analyse.

**Bijzonderheden:** 4 policies met sterk verschillende prestaties. Expert policy presteert best. Random policy toont veel constraint violations. State-action-reward-next_state formaat voor directe RL-toepassing.

---

### 35. Transfer Learning Reactoren (`transfer_learning_reactoren.csv`)

**Context:** Dezelfde reactor op 2 locaties (Plant A: 2000 samples, Plant B: 100 samples). Subtiel verschillende proceskarakteristieken (domain shift).

**Kolommen:**
- `sample_id`, `plant` - identificatie (Plant_A / Plant_B)
- `temperatuur_C`, `druk_bar`, `debiet_kgh`, `katalysator_kgh`, `voeding_conc_molL` - procesparameters
- `conversie` - **target (regressie):** conversiegraad
- `selectiviteit` - **target (regressie):** selectiviteit
- `opbrengst_pct` - conversie × selectiviteit × 100
- `energie_kWh` - energieverbruik
- `heeft_label` - 1 = gelabeld (100% Plant A, 30% Plant B)

**ML-toepassingen:** Domain adaptation, transfer learning, few-shot learning, domain-invariant features, model fine-tuning met beperkte data.

**Bijzonderheden:** Plant B heeft hogere baseline conversie maar lagere selectiviteit. Slechts 100 samples (30% gelabeld) voor target domain. Niet-lineaire relaties met subtiele domeinverschuiving.

---

### 36. Active Learning Pool (`active_learning_pool.csv`)

**Context:** 5000 procesmonsters waarvan slechts 50 (1%) gelabeld. 4 verborgen regimes in de data. Inclusief model-uncertainty scores.

**Kolommen:**
- `sample_id` - sample identifier
- `temperatuur_C`, `druk_bar`, `debiet_Lh`, `pH`, `conductiviteit_mScm`, `viscositeit_mPas`, `turbiditeit_NTU`, `opgeloste_O2_mgL` - 8 procesfeatures
- `kwaliteitsscore` - **target (regressie):** productkwaliteit (0-100)
- `is_anomalie` - 1 = anomalie (5%)
- `regime` - verborgen operatieregime (0-3)
- `is_gelabeld` - 1 = gelabeld (slechts 1%)
- `model_voorspelling` - voorspelling van initieel model
- `model_onzekerheid` - **target (query strategie):** modelonzekerheid
- `informativiteit` - informativeness score voor sample selectie

**ML-toepassingen:** Pool-based active learning, uncertainty sampling, query-by-committee, semi-supervised learning, label-efficiënte modellering.

**Bijzonderheden:** 4 operatieregimes met verschillende kwaliteitsmodellen. 5% anomalieën. Hoge model-onzekerheid bij regime-grenzen. Slechts 50 van 5000 gelabeld.

---

### 37. NLP Onderhoudslogboek (`nlp_onderhoudslogboek.csv`)

**Context:** 3000 onderhoudswerkorders over 2 jaar met gestructureerde velden en vrije-tekst beschrijvingen. 4 equipment systemen, 7+ faalmodi.

**Kolommen:**
- `werkorder_id` - werkorder identifier
- `datum` - datum
- `tag` - equipment tag (bijv. PMP-342)
- `equipment_type` - pomp / compressor / motor / klep / sensor / etc.
- `systeem` - mechanisch / elektrisch / instrumentatie / piping
- `locatie` - Unit_100 / Unit_200 / etc.
- `werkorder_type` - correctief / preventief / predictief / verbetering / inspectie
- `prioriteit` - **target (classificatie):** laag / medium / hoog / kritiek
- `faalmodus` - **target (NER/classificatie):** specifieke faalmodus
- `beschrijving` - **input (NLP):** vrije tekst met symptoom + actie
- `responstijd_uur`, `reparatietijd_uur`, `stilstandtijd_uur` - tijden
- `materiaalkosten_EUR`, `arbeidskosten_EUR`, `totaalkosten_EUR` - kosten
- `terugkerend` - 1 = terugkerend probleem (15%)

**ML-toepassingen:** Tekstclassificatie, named entity recognition, MTBF/MTTR analyse, kostenvoorspelling, recurring failure detectie, prioriteitsvoorspelling.

**Bijzonderheden:** Realistische werkorderteksten in het Nederlands. Mix van correctief (35%) en preventief (30%) onderhoud. Recurring failures gemarkeerd.

---

### 38. Virtual Metrology (`virtual_metrology.csv`)

**Context:** Procesdata met dure/trage lab-metingen (slechts 10% beschikbaar) en goedkope inline metingen. 3 kamers met subtiele verschillen. Concept drift door onderhoudscycli.

**Kolommen:**
- `sample_id` - sample identifier
- `kamer` - Chamber_A / B / C (elk met eigen offset)
- `positie` - center / edge / corner
- `onderhoudscyclus` - cyclus sinds laatste onderhoud (1-50)
- `temp_zone1_C` t/m `temp_zone3_C`, `druk_mbar`, `gasflow_1_sccm`, `gasflow_2_sccm` - procesdata (altijd beschikbaar)
- `vermogen_W`, `procestijd_min`, `kamer_vochtigheid_pct`, `substraat_temp_C` - aanvullende procesdata
- `inline_dikte_nm`, `inline_reflectantie` - inline metingen (altijd, maar minder nauwkeurig)
- `lab_dikte_nm` - **target (regressie):** nauwkeurige diktemeting (slechts 10% beschikbaar)
- `lab_uniformiteit_pct` - **target (regressie):** uniformiteit (10%)
- `lab_stress_MPa` - **target (regressie):** stress (10%)
- `lab_gemeten` - 1 = lab meting beschikbaar

**ML-toepassingen:** Semi-supervised regressie, missing data handling, virtual metrology, multi-output prediction, concept drift (kamer + onderhoud).

**Bijzonderheden:** 90% ontbrekende labels. Kamer-specifieke offsets. Geleidelijke drift door onderhoudscycli. Inline metingen als proxy features met meer ruis.

---

### 39. Hot-Melt Extrusie (`extrusie_hotmelt.csv`)

**Context:** Twin-screw farmaceutische hot-melt extruder met 8 barrel zones. Inline Raman spectroscopie. Data elke 5 seconden, 24 uur.

**Kolommen:**
- `timestamp` - tijdstip
- `schroefsnelheid_RPM` - schroeftoerental
- `voeding_kgh` - voedingssnelheid (kg/h)
- `torque_pct` - schroef torque (% van max)
- `SME_kWhkg` - specific mechanical energy (kWh/kg)
- `die_druk_bar` - druk aan de die
- `smelt_temp_C` - smelttemperatuur
- `zone1_temp_C` t/m `zone8_temp_C` - barrel zone temperaturen (actueel)
- `zone1_sp_C` t/m `zone8_sp_C` - barrel zone setpoints
- `API_gehalte_pct` - **target (soft sensor):** API-gehalte
- `degradatie_pct` - **target (regressie):** degradatiepercentage
- `Raman_API_pct` - inline Raman meting (elke 30 sec, anders leeg)
- `Raman_gemeten` - 1 = Raman meting beschikbaar
- `event` - normaal / feeder_puls / heater_fout / materiaal_batch_verschil

**ML-toepassingen:** Soft sensor (API-gehalte uit procesdata), Raman kalibratie, residence time modellering, procesverstoringen detectie.

**Bijzonderheden:** 8 temperatuurzones met thermische interactie. SME als procesindikator. Raman als referentiemeting (elke 30 sec). 3 procesverstoringen.

---

### 40. Membraanfiltratie (`membraan_filtratie.csv`)

**Context:** UF/RO/NF membraanfiltratie voor waterzuivering. 4 modules, 6 maanden per uur. Fouling opbouw met periodieke reinigingscycli.

**Kolommen:**
- `timestamp` - tijdstip
- `module_id` - module identifier (MEM-01 t/m MEM-04)
- `module_type` - RO / UF / NF
- `voeding_TDS_mgL`, `voeding_turbiditeit_NTU`, `voeding_temp_C`, `voeding_pH` - voedingskwaliteit
- `voeding_druk_bar` - voedingsdruk
- `TMP_bar` - **feature:** transmembraandruk (stijgt met fouling)
- `flux_Lm2h` - **target (regressie):** permeaatflux (L/m²/h)
- `specifieke_flux_Lm2hbar` - flux genormaliseerd op druk
- `permeaat_TDS_mgL`, `permeaat_conductiviteit_mScm` - permeaatkwaliteit
- `recovery` - waterterugwinning (fractie)
- `fouling_weerstand` - **target (regressie):** fouling factor
- `SEC_kWhm3` - specifiek energieverbruik (kWh/m³)
- `is_reiniging` - 1 = tijdens reinigingscyclus
- `fouling_fase` - **target (classificatie):** schoon / lichte_fouling / matige_fouling / ernstige_fouling / reiniging

**ML-toepassingen:** Fouling voorspelling, reinigingsmoment optimalisatie, flux modellering, seizoenseffect analyse, vergelijking membraantypes.

**Bijzonderheden:** 3 membraantypes (RO/UF/NF) met verschillende fouling-patronen. Seizoenseffect op voedingskwaliteit. Periodieke reinigingscycli (~30 dagen). Biologische, scaling en particulaire fouling componenten.

---

## Gebruik

```bash
# Genereer alle datasets opnieuw
python generate_datasets.py

# Gebruik in Python
import pandas as pd
df = pd.read_csv("data/batch_reactor_yield.csv")
```

## Moeilijkheidsgraad per ML-techniek

| Techniek | Aanbevolen datasets |
|----------|-------------------|
| Lineaire regressie | 1, 3, 4, 19 |
| Niet-lineaire regressie | 1, 6, 11, 12, 18, 33 |
| Binaire classificatie | 3, 7, 11, 15, 22, 24, 27, 33 |
| Multiclass classificatie | 1, 5, 8, 9, 14, 17, 20, 21, 24, 26, 28, 37, 40 |
| Clustering | 4, 9, 13, 22, 36 |
| Tijdreeksanalyse | 2, 5, 7, 10, 12, 13, 16, 17, 18, 23, 27, 31, 39, 40 |
| Anomalie-detectie | 2, 8, 9, 13, 15, 16, 22, 24, 28, 31, 36 |
| Predictive maintenance | 5, 12, 17, 26, 40 |
| Procesoptimalisatie | 1, 6, 7, 14, 19, 23, 27, 30, 33 |
| Batch trajectory modelling | 10, 14, 18, 20, 29 |
| Soft sensor ontwikkeling | 13, 14, 15, 16, 18, 38, 39 |
| Multivariate procesmonitoring (PCA/PLS) | 13, 15, 16, 17, 21, 25, 29 |
| Cascade fault detection | 16 |
| Root cause analysis | 13, 15, 16, 22 |
| Grade-transitie optimalisatie | 13 |
| Bioprocessen / celcultuur | 10, 18, 19, 20 |
| Control loop performance | 21, 30 |
| Alarmbeheer / alarm rationalisatie | 22 |
| Energievoorspelling / -optimalisatie | 23 |
| Degradatiekinetiek | 11, 19 |
| Eindpuntdetectie | 14, 20, 27 |
| Machine vision / beeldinspectie | 24 |
| Spectroscopie / multivariate kalibratie | 25 |
| Smart valve diagnostiek | 26 |
| GMP/cleanroom monitoring | 27, 28 |
| Golden batch / MPCA | 29 |
| Model Predictive Control analyse | 30 |
| Digital twin / concept drift | 31 |
| NLP / text mining | 32, 37 |
| Multi-product receptoptimalisatie | 33 |
| Reinforcement learning | 34 |
| Transfer learning / domeinadaptatie | 35 |
| Active learning / semi-supervised | 36, 38 |
| Virtual metrology / missing labels | 38 |
| Extrusie / PAT (Raman) | 39 |
| Membraanfiltratie / fouling | 40 |

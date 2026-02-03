#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ГЕНЕТИК СИНДРОМЛАР ХАВФ БАХОЛАШ ДАСТУРИ
DELFIA Revvity реагентлари асосида
Версия 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import math
import warnings
warnings.filterwarnings('ignore')

# ==================== СЕССИЯ СОЗЛАМАЛАРИ ====================
if 'screening_type' not in st.session_state:
    st.session_state.screening_type = "first"
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {}
if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 1

# ==================== ЎЗГАРМАСЛАР ВА НОРМАЛАР ====================

# Генетик синдромлар учун асосий хавфлар (1:N)
BASE_RISKS = {
    'downs': 1/800,      # Даун синдроми (Трисомия 21)
    'edwards': 1/3000,   # Эдвардс синдроми (Трисомия 18)
    'patau': 1/5000,     # Патау синдроми (Трисомия 13)
    'turner': 1/2500,    # Тернер синдроми (45,X)
    'ntd': 1/1000        # Нейротубуляр дефект
}

# Ёш бўйича хавф кўпайтирувчилари
AGE_RISK_MULTIPLIERS = {
    20: {'downs': 0.5, 'edwards': 0.3, 'patau': 0.3, 'turner': 0.4},
    25: {'downs': 0.7, 'edwards': 0.5, 'patau': 0.5, 'turner': 0.6},
    30: {'downs': 1.0, 'edwards': 1.0, 'patau': 1.0, 'turner': 1.0},
    35: {'downs': 2.5, 'edwards': 3.0, 'patau': 3.5, 'turner': 2.0},
    40: {'downs': 5.0, 'edwards': 8.0, 'patau': 10.0, 'turner': 4.0},
    45: {'downs': 10.0, 'edwards': 15.0, 'patau': 20.0, 'turner': 8.0}
}

# DELFIA Revvity биринчи триместр нормалари
DELFIA_FIRST_TRIMESTER_NORMS = {
    'PAPP_A': {
        'unit': 'U/L',
        'median_values': {
            10: 1.0, 11: 1.2, 12: 1.4, 13: 1.6, 14: 1.8
        },
        'MoM_low': 0.4,
        'MoM_high': 2.5,
        'weight_correction': True
    },
    'FREE_BETA_HCG': {
        'unit': 'ng/ml',
        'median_values': {
            10: 40.0, 11: 60.0, 12: 80.0, 13: 100.0, 14: 120.0
        },
        'MoM_low': 0.5,
        'MoM_high': 2.0,
        'weight_correction': True
    },
    'NT': {
        'unit': 'мм',
        'median_values': {
            10: 1.2, 11: 1.3, 12: 1.4, 13: 1.5, 14: 1.5
        },
        'MoM_low': 0.8,
        'MoM_high': 2.0,
        'cutoff': 2.5,  # NT катталиги чегараси
        'weight_correction': False
    }
}

# DELFIA Revvity иккинчи триместр нормалари
DELFIA_SECOND_TRIMESTER_NORMS = {
    'AFP': {
        'unit': 'ng/ml',
        'median_values': {
            15: 30.0, 16: 35.0, 17: 40.0, 18: 45.0, 19: 50.0, 20: 55.0
        },
        'MoM_low': 0.5,
        'MoM_high': 2.0,
        'weight_correction': True
    },
    'TOTAL_HCG': {
        'unit': 'IU/L',
        'median_values': {
            15: 30000, 16: 28000, 17: 25000, 18: 22000, 19: 20000, 20: 18000
        },
        'MoM_low': 0.5,
        'MoM_high': 2.0,
        'weight_correction': True
    },
    'UE3': {
        'unit': 'nmol/L',
        'median_values': {
            15: 2.5, 16: 3.0, 17: 3.5, 18: 4.0, 19: 4.5, 20: 5.0
        },
        'MoM_low': 0.5,
        'MoM_high': 2.0,
        'weight_correction': True
    }
}

# Синдромлар тавсифи
SYNDROME_DESCRIPTIONS = {
    'downs': {
        'name': 'Даун синдроми',
        'scientific': 'Трисомия 21',
        'description': 'Интеллектуал нотўликлик, юрак аномалиялари, мускул гипотонияси',
        'risk_factors': ['Ҳар иккала ота-онада ёш', 'Оилда борилиги', 'Диабет'],
        'color': '#ff6b6b',
        'icon': '👶'
    },
    'edwards': {
        'name': 'Эдвардс синдроми',
        'scientific': 'Трисомия 18',
        'description': 'Оғир кўп орган зарарланиши, йўл-йўлақа аномалиялари',
        'risk_factors': ['Онанинг ёши', 'Қийин вазн орттириш'],
        'color': '#ff9800',
        'icon': '⚠️'
    },
    'patau': {
        'name': 'Патау синдроми',
        'scientific': 'Трисомия 13',
        'description': 'Неврологик аномалиялар, кўз ва юз аномалиялари',
        'risk_factors': ['Ота-она ёши', 'Радиацияга мулоқот'],
        'color': '#ff5722',
        'icon': '🔬'
    },
    'turner': {
        'name': 'Тернер синдроми',
        'scientific': '45,X',
        'description': 'Бўй пастлиги, жинсий руксатсизлик, юрак аномалиялари',
        'risk_factors': ['Отанинг ёши', 'Модда алмашинуви'],
        'color': '#9c27b0',
        'icon': '🧬'
    },
    'ntd': {
        'name': 'Нейротубуляр дефект',
        'scientific': 'НТД',
        'description': 'Спина бифида, анэнцефалия, менингоцеле',
        'risk_factors': ['Фолат етишмовчилиги', 'Диабет', 'Ожирение'],
        'color': '#4caf50',
        'icon': '📏'
    }
}

# ==================== ФУНКЦИЯЛАР ====================

def calculate_bmi(weight_kg, height_cm):
    """Body Mass Index (BMI) ҳисоблаш"""
    if height_cm > 0:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 1)
    return 22.0

def get_bmi_category(bmi):
    """BMI категориясини аниқлаш"""
    if bmi < 18.5:
        return "Паст вазн", "bmi-low"
    elif 18.5 <= bmi < 25:
        return "Нормал", "bmi-normal"
    elif 25 <= bmi < 30:
        return "Ортиқча вазн", "bmi-overweight"
    else:
        return "Семизлик", "bmi-obese"

def get_median_value(parameter, gestational_week, trimester="first"):
    """Гестацион ҳафтага кўра медиана қийматини олиш"""
    if trimester == "first":
        norms = DELFIA_FIRST_TRIMESTER_NORMS
    else:
        norms = DELFIA_SECOND_TRIMESTER_NORMS
    
    if parameter in norms:
        weeks = list(norms[parameter]['median_values'].keys())
        
        if gestational_week in norms[parameter]['median_values']:
            return norms[parameter]['median_values'][gestational_week]
        
        # Энг яқин ҳафтани топиш
        closest_week = min(weeks, key=lambda x: abs(x - gestational_week))
        return norms[parameter]['median_values'][closest_week]
    
    return 1.0

def calculate_mom_value(measured_value, parameter, gestational_week, maternal_weight=None, trimester="first"):
    """Multiple of Median (MoM) қийматини ҳисоблаш"""
    median = get_median_value(parameter, gestational_week, trimester)
    
    if median <= 0:
        return 1.0
    
    # Асосий MoM ҳисоблаш
    mom = measured_value / median
    
    # Вазна коррекцияси (агар зарур бўлса)
    if maternal_weight and trimester == "first":
        norms = DELFIA_FIRST_TRIMESTER_NORMS if trimester == "first" else DELFIA_SECOND_TRIMESTER_NORMS
        if parameter in norms and norms[parameter].get('weight_correction', False):
            # Стандарт вазн 65 кг деб ҳисобланади
            weight_correction = math.sqrt(maternal_weight / 65.0)
            mom = mom / weight_correction
    
    return round(mom, 2)

def get_age_risk_multiplier(age, syndrome):
    """Ёш бўйича хавф кўпайтирувчисини олиш"""
    ages = sorted(AGE_RISK_MULTIPLIERS.keys())
    
    if age <= ages[0]:
        return AGE_RISK_MULTIPLIERS[ages[0]][syndrome]
    elif age >= ages[-1]:
        return AGE_RISK_MULTIPLIERS[ages[-1]][syndrome]
    
    # Интерполяция қилиш
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            age1, age2 = ages[i], ages[i + 1]
            mult1 = AGE_RISK_MULTIPLIERS[age1][syndrome]
            mult2 = AGE_RISK_MULTIPLIERS[age2][syndrome]
            
            # Чизиқли интерполяция
            interpolation_factor = (age - age1) / (age2 - age1)
            risk_multiplier = mult1 + interpolation_factor * (mult2 - mult1)
            return round(risk_multiplier, 2)
    
    return 1.0

def calculate_syndrome_risks(patient_age, marker_moms, trimester="first"):
    """
    Барча генетик синдромлар учун хавфларни ҳисоблаш
    """
    risks = {}
    
    # Маркер MoM қийматлари
    nt_mom = marker_moms.get('nt_mom', 1.0)
    papp_mom = marker_moms.get('papp_mom', 1.0)
    hcg_mom = marker_moms.get('hcg_mom', 1.0)
    afp_mom = marker_moms.get('afp_mom', 1.0)
    total_hcg_mom = marker_moms.get('total_hcg_mom', 1.0)
    ue3_mom = marker_moms.get('ue3_mom', 1.0)
    
    # 1. ЁШ ХАВФЛАРИНИ ҲИСОБЛАШ
    age_risks = {}
    for syndrome in ['downs', 'edwards', 'patau', 'turner']:
        age_risks[syndrome] = get_age_risk_multiplier(patient_age, syndrome)
    
    # 2. ДАУН СИНДРОМИ ХАВФИ
    base_down_risk = BASE_RISKS['downs']
    down_risk = base_down_risk * age_risks['downs']
    
    # PAPP-A коррекцияси
    if papp_mom < 0.3:
        down_risk *= 3.0
    elif papp_mom < 0.4:
        down_risk *= 2.0
    elif papp_mom < 0.5:
        down_risk *= 1.5
    elif papp_mom > 2.5:
        down_risk *= 1.2
    
    # Free β-hCG коррекцияси
    if hcg_mom < 0.2:
        down_risk *= 2.5
    elif hcg_mom < 0.3:
        down_risk *= 1.8
    elif hcg_mom > 2.5:
        down_risk *= 2.0
    elif hcg_mom > 3.5:
        down_risk *= 2.5
    
    # NT коррекцияси
    if nt_mom < 0.6:
        down_risk *= 0.7
    elif nt_mom < 0.8:
        down_risk *= 0.8
    elif nt_mom > 2.0:
        down_risk *= 3.0
    elif nt_mom > 3.0:
        down_risk *= 5.0
    
    risks['downs'] = min(down_risk, 0.5)  # Максимум 50% хавф
    
    # 3. ЭДВАРДС СИНДРОМИ ХАВФИ
    edwards_risk = BASE_RISKS['edwards'] * age_risks['edwards']
    
    if papp_mom < 0.2:
        edwards_risk *= 4.0
    elif papp_mom < 0.3:
        edwards_risk *= 2.5
    
    if hcg_mom < 0.1:
        edwards_risk *= 3.0
    elif hcg_mom < 0.2:
        edwards_risk *= 2.0
    
    if nt_mom > 2.5:
        edwards_risk *= 4.0
    
    risks['edwards'] = min(edwards_risk, 0.5)
    
    # 4. ПАТАУ СИНДРОМИ ХАВФИ
    patau_risk = BASE_RISKS['patau'] * age_risks['patau']
    
    if papp_mom < 0.2:
        patau_risk *= 5.0
    elif papp_mom < 0.3:
        patau_risk *= 3.0
    
    if hcg_mom < 0.15:
        patau_risk *= 3.5
    elif hcg_mom < 0.25:
        patau_risk *= 2.5
    
    if nt_mom > 2.8:
        patau_risk *= 5.0
    
    risks['patau'] = min(patau_risk, 0.5)
    
    # 5. ТЕРНЕР СИНДРОМИ ХАВФИ
    turner_risk = BASE_RISKS['turner'] * age_risks['turner']
    
    if hcg_mom > 2.0:
        turner_risk *= 2.0
    elif hcg_mom > 3.0:
        turner_risk *= 3.0
    
    if nt_mom > 3.0:
        turner_risk *= 4.0
    
    risks['turner'] = min(turner_risk, 0.5)
    
    # 6. НТД ХАВФИ
    ntd_risk = BASE_RISKS['ntd']
    
    if afp_mom > 2.5:
        ntd_risk = 0.01  # 1:100
    elif afp_mom > 2.0:
        ntd_risk = 0.02  # 1:50
    elif afp_mom < 0.5:
        ntd_risk = ntd_risk * 0.7  # Паст AFP - хавф камайиши
    
    risks['ntd'] = min(ntd_risk, 0.5)
    
    # 7. ИККИЛАМЧИ СКРИНИНГ КОРРЕКЦИЯСИ
    if trimester == "second" and all([afp_mom, total_hcg_mom, ue3_mom]):
        quad_correction = 1.0
        
        # AFP коррекцияси
        if afp_mom < 0.5:
            quad_correction *= 0.8
        elif afp_mom > 2.0:
            quad_correction *= 1.3
        
        # Total hCG коррекцияси
        if total_hcg_mom < 0.5:
            quad_correction *= 0.9
        elif total_hcg_mom > 2.0:
            quad_correction *= 1.8
        
        # uE3 коррекцияси
        if ue3_mom < 0.5:
            quad_correction *= 1.5
        
        # Хавфларга коррекция қўллаш
        risks['downs'] *= quad_correction
        risks['edwards'] *= quad_correction * 1.2
        risks['patau'] *= quad_correction * 1.3
    
    # 8. ЁШ ХАВФЛАРИНИ САҚЛАШ
    risks['age_risk'] = age_risks
    
    return risks

def get_risk_category(risk_value):
    """Хавф қийматига кўра категория аниқлаш"""
    if risk_value <= 0:
        return "НОМАЪЛУМ", "risk-unknown", "#9e9e9e"
    elif risk_value > 0.1:      # 1:10 дан юқори
        return "КРИТИК", "risk-critical", "#b71c1c"
    elif risk_value > 0.05:     # 1:20
        return "ЖУДА ЮҚОРИ", "risk-high", "#e65100"
    elif risk_value > 0.02:     # 1:50
        return "ЮҚОРИ", "risk-high", "#f57c00"
    elif risk_value > 0.01:     # 1:100
        return "ЎРТАЧА-ЮҚОРИ", "risk-medium", "#f57f17"
    elif risk_value > 0.005:    # 1:200
        return "ЎРТАЧА", "risk-medium", "#f9a825"
    elif risk_value > 0.001:    # 1:1000
        return "ПАСТ-ЎРТАЧА", "risk-low", "#388e3c"
    else:                       # 1:1000 дан паст
        return "ПАСТ", "risk-low", "#1b5e20"

def format_risk_display(risk_value):
    """Хавф қийматини кўринишли форматда кўрсатиш"""
    if risk_value <= 0:
        return "1:∞"
    
    try:
        ratio = int(1 / risk_value)
        return f"1:{ratio:,}".replace(",", " ")
    except:
        return f"1:{int(1/risk_value)}"

def save_patient_record(patient_data):
    """Бемор маълумотларини сақлаш"""
    try:
        # Пациент ID генерацияси
        patient_id = f"PAT-{datetime.now().strftime('%Y%m%d')}-{st.session_state.patient_counter:03d}"
        st.session_state.patient_counter += 1
        
        patient_data['patient_id'] = patient_id
        patient_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.session_state.patient_history.append(patient_data)
        
        # Фақат охирги 20 та маълумотни сақлаш
        if len(st.session_state.patient_history) > 20:
            st.session_state.patient_history = st.session_state.patient_history[-20:]
        
        return patient_id
    except Exception as e:
        st.error(f"Сақлашда хатолик: {str(e)}")
        return None

def get_patient_summary():
    """Беморлар тарихини қисқача кўрсатиш"""
    if not st.session_state.patient_history:
        return None
    
    summary = []
    for patient in st.session_state.patient_history[-5:][::-1]:  # Охирги 5 таси
        summary.append({
            'name': patient.get('name', 'Номаълум'),
            'age': patient.get('age', 30),
            'gestational_age': patient.get('gestational_age', 12),
            'screening_type': patient.get('screening_type', 'first'),
            'timestamp': patient.get('timestamp', ''),
            'downs_risk': patient.get('risks', {}).get('downs', 0)
        })
    
    return summary

# ==================== CSS СТИЛЛАР ====================
PAGE_CSS = """
<style>
/* Асосий сарлавҳа */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #0d47a1, #1565c0, #1976d2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 20px 0;
    padding: 15px;
    border-radius: 15px;
    border: 3px solid #bbdefb;
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.15);
}

.sub-title {
    font-size: 1.4rem;
    text-align: center;
    color: #1565c0;
    margin-bottom: 30px;
    padding: 15px;
    background: linear-gradient(90deg, #e3f2fd, #bbdefb);
    border-radius: 12px;
    border: 2px solid #90caf9;
}

/* Скрининг тугмалари */
.screening-btn {
    font-size: 1.1rem;
    font-weight: 600;
    padding: 15px;
    border-radius: 10px;
    transition: all 0.3s ease;
    margin: 5px 0;
}

.screening-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* Синдром карталари */
.syndrome-card {
    padding: 20px;
    border-radius: 15px;
    margin: 15px 0;
    border: 3px solid;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}

.syndrome-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.downs-card { border-color: #ff6b6b; background: linear-gradient(135deg, #ffebee, #ffcdd2); }
.edwards-card { border-color: #ff9800; background: linear-gradient(135deg, #fff3e0, #ffe0b2); }
.patau-card { border-color: #ff5722; background: linear-gradient(135deg, #fbe9e7, #ffccbc); }
.turner-card { border-color: #9c27b0; background: linear-gradient(135deg, #f3e5f5, #e1bee7); }
.ntd-card { border-color: #4caf50; background: linear-gradient(135deg, #e8f5e9, #c8e6c9); }

/* Хавф категориялари */
.risk-critical {
    background: linear-gradient(135deg, #b71c1c, #d32f2f);
    color: white;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    display: inline-block;
    border: 3px solid #ff5252;
    box-shadow: 0 6px 20px rgba(183, 28, 28, 0.3);
    animation: pulse 2s infinite;
    font-size: 1.1rem;
}

.risk-high {
    background: linear-gradient(135deg, #e65100, #f57c00);
    color: white;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    display: inline-block;
    border: 3px solid #ffb74d;
    box-shadow: 0 6px 18px rgba(230, 81, 0, 0.3);
    font-size: 1.1rem;
}

.risk-medium {
    background: linear-gradient(135deg, #f57f17, #f9a825);
    color: #333;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    display: inline-block;
    border: 3px solid #ffd54f;
    box-shadow: 0 6px 16px rgba(245, 127, 23, 0.3);
    font-size: 1.1rem;
}

.risk-low {
    background: linear-gradient(135deg, #1b5e20, #388e3c);
    color: white;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    display: inline-block;
    border: 3px solid #66bb6a;
    box-shadow: 0 6px 16px rgba(27, 94, 32, 0.3);
    font-size: 1.1rem;
}

.risk-unknown {
    background: linear-gradient(135deg, #616161, #9e9e9e);
    color: white;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    display: inline-block;
    border: 3px solid #bdbdbd;
    font-size: 1.1rem;
}

/* Анимация */
@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(183, 28, 28, 0.7); }
    50% { transform: scale(1.05); }
    70% { box-shadow: 0 0 0 15px rgba(183, 28, 28, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(183, 28, 28, 0); }
}

/* Метрика карталари */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    margin: 10px 0;
    border-left: 5px solid #2196f3;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* Инфо блоки */
.info-box {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #90caf9;
    margin: 20px 0;
}

/* Тавсия блоки */
.recommendation-box {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #ffd54f;
    margin: 20px 0;
}

/* Хавфсизлик ёзуви */
.warning-box {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #ff5252;
    margin: 20px 0;
    color: #c62828;
}

/* BMI категориялари */
.bmi-low { color: #0277bd; }
.bmi-normal { color: #2e7d32; }
.bmi-overweight { color: #f57c00; }
.bmi-obese { color: #c62828; }
</style>
"""

# ==================== САХИФА КОНФИГУРАЦИЯСИ ====================
st.set_page_config(
    page_title="Генетик Синдромлар Хавф Бахолаш Дастури",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Tolibjon-code/ok-genetic-risk-app',
        'Report a bug': 'https://github.com/Tolibjon-code/ok-genetic-risk-app/issues',
        'About': "### Генетик Синдромлар Хавф Бахолаш Дастури\n\nDELFIA Revvity реагентлари асосида\n\nВерсия 1.0.0"
    }
)

# CSS стилларни қўшиш
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ==================== САРЛАВҲА ====================
st.markdown('<h1 class="main-title">🧬 ГЕНЕТИК СИНДРОМЛАР ХАВФ БАХОЛАШ ДАСТУРИ</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Даун • Эдвардс • Патау • Тернер • НТД | DELFIA Revvity асосида</p>', unsafe_allow_html=True)

# ==================== СКРИНИНГ ТУРИ ТАНЛАШ ====================
st.markdown("### 📋 Скрининг турини танланг")

col1, col2 = st.columns(2)
with col1:
    if st.button(
        f"👶 **БИРИНЧИ СКРИНИНГ**\n(10-14 ҳафта)\nNT, PAPP-A, Free β-hCG", 
        use_container_width=True,
        type="primary" if st.session_state.screening_type == "first" else "secondary",
        help="Биринчи триместр скрининги - NT, PAPP-A, Free β-hCG"
    ):
        st.session_state.screening_type = "first"
        st.rerun()

with col2:
    if st.button(
        f"🤰 **ИККИЛАМЧИ СКРИНИНГ**\n(15-20 ҳафта)\nAFP, Total hCG, uE3", 
        use_container_width=True,
        type="primary" if st.session_state.screening_type == "second" else "secondary",
        help="Иккинчи триместр скрининги - AFP, Total hCG, uE3"
    ):
        st.session_state.screening_type = "second"
        st.rerun()

st.markdown("---")

# ==================== САЙДБАР - БЕМОР МАЪЛУМОТЛАРИ ====================
with st.sidebar:
    st.markdown(f"### {SYNDROME_DESCRIPTIONS['downs']['icon']} БЕМОР МАЪЛУМОТЛАРИ")
    
    # Бемор исми
    patient_name = st.text_input(
        "**Фамилия Исм Шариф**",
        placeholder="Мадина Алиева",
        help="Беморнинг тўлиқ исми"
    )
    
    # Ёш ва хомилалик ҳафтаси
    col_age, col_week = st.columns(2)
    with col_age:
        patient_age = st.number_input(
            "**Ёши**", 
            min_value=15, 
            max_value=55, 
            value=30,
            help="Беморнинг ёши (15-55)"
        )
    
    with col_week:
        if st.session_state.screening_type == "first":
            gestational_age = st.number_input(
                "**Хомилалик (ҳафта)**",
                min_value=10,
                max_value=14,
                value=12,
                help="Гестацион ҳафта (10-14)"
            )
        else:
            gestational_age = st.number_input(
                "**Хомилалик (ҳафта)**",
                min_value=15,
                max_value=20,
                value=18,
                help="Гестацион ҳафта (15-20)"
            )
    
    # Бўй ва вазн
    col_height, col_weight = st.columns(2)
    with col_height:
        height = st.number_input(
            "**Бўй (см)**",
            min_value=140,
            max_value=200,
            value=165,
            help="Беморнинг бўйи"
        )
    
    with col_weight:
        weight = st.number_input(
            "**Вазн (кг)**",
            min_value=40,
            max_value=150,
            value=65,
            help="Беморнинг вазни"
        )
    
    # BMI ҳисоблаш ва кўрсатиш
    if height > 0:
        bmi = calculate_bmi(weight, height)
        bmi_category, bmi_class = get_bmi_category(bmi)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666;">📊 BODY MASS INDEX</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{bmi:.1f}</div>
            <div class="{bmi_class}" style="font-weight: bold;">{bmi_category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # СКРИНИНГ ПАРАМЕТРЛАРИ
    if st.session_state.screening_type == "first":
        st.markdown(f"### {SYNDROME_DESCRIPTIONS['edwards']['icon']} БИРИНЧИ СКРИНИНГ ПАРАМЕТРЛАРИ")
        
        # NT қалинлиги
        nt_value = st.slider(
            "**NT қалинлиги (мм)**",
            min_value=0.5,
            max_value=5.0,
            value=1.8,
            step=0.1,
            help="Нухал транспаренси қалинлиги (норма: 0.8-2.5 мм)"
        )
        
        # PAPP-A
        papp_a_value = st.number_input(
            "**PAPP-A Қиймати (U/L)**",
            min_value=0.1,
            max_value=10.0,
            value=1.4,
            step=0.1,
            help="Pregnancy-associated plasma protein A"
        )
        
        # Free β-hCG
        free_beta_hcg_value = st.number_input(
            "**Free β-hCG Қиймати (ng/ml)**",
            min_value=1.0,
            max_value=300.0,
            value=80.0,
            step=1.0,
            help="Free beta human chorionic gonadotropin"
        )
        
    else:  # Иккиламчи скрининг
        st.markdown(f"### {SYNDROME_DESCRIPTIONS['patau']['icon']} ИККИЛАМЧИ СКРИНИНГ ПАРАМЕТРЛАРИ")
        
        # AFP
        afp_value = st.number_input(
            "**AFP Қиймати (ng/ml)**",
            min_value=1.0,
            max_value=200.0,
            value=45.0,
            step=1.0,
            help="Alpha-fetoprotein"
        )
        
        # Total hCG
        total_hcg_value = st.number_input(
            "**Total hCG Қиймати (IU/L)**",
            min_value=1000,
            max_value=100000,
            value=22000,
            step=1000,
            help="Total human chorionic gonadotropin"
        )
        
        # uE3
        ue3_value = st.number_input(
            "**uE3 Қиймати (nmol/L)**",
            min_value=0.1,
            max_value=20.0,
            value=4.0,
            step=0.1,
            help="Unconjugated estriol"
        )
    
    st.markdown("---")
    
    # ҲИСОБЛАШ ТУГМАСИ
    calculate_btn = st.button(
        f"🧬 **ГЕНЕТИК ХАВФЛАРНИ ҲИСОБЛАШ**",
        type="primary",
        use_container_width=True,
        use_container_width=True,
        help="Барча параметрлар асосида генетик хавфларни ҳисоблаш"
    )

# ==================== АСОСИЙ КОНТЕНТ ====================

if calculate_btn:
    # Валидация
    if not patient_name or patient_name.strip() == "":
        st.error("❌ **ХАТО:** Илтимос, беморнинг исмини киритинг!")
        st.stop()
    
    if height <= 0:
        st.error("❌ **ХАТО:** Бўй қиймати нотўғри!")
        st.stop()
    
    # BMI ҳисоблаш
    bmi = calculate_bmi(weight, height)
    bmi_category, _ = get_bmi_category(bmi)
    
    with st.spinner(f"**{patient_name}** учун генетик хавфлар ҳисобланади..."):
        try:
            # MoM қийматларини ҳисоблаш
            marker_moms = {}
            
            if st.session_state.screening_type == "first":
                # Биринчи скрининг MoM қийматлари
                nt_mom = calculate_mom_value(nt_value, 'NT', gestational_age, weight, "first")
                papp_mom = calculate_mom_value(papp_a_value, 'PAPP_A', gestational_age, weight, "first")
                hcg_mom = calculate_mom_value(free_beta_hcg_value, 'FREE_BETA_HCG', gestational_age, weight, "first")
                
                marker_moms = {
                    'nt_mom': nt_mom,
                    'papp_mom': papp_mom,
                    'hcg_mom': hcg_mom
                }
                
                risks = calculate_syndrome_risks(patient_age, marker_moms, "first")
                
                # Бемор маълумотларини тузиш
                patient_data = {
                    'name': patient_name,
                    'age': patient_age,
                    'gestational_age': gestational_age,
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'bmi_category': bmi_category,
                    'screening_type': 'first',
                    'parameters': {
                        'nt': nt_value,
                        'nt_mom': nt_mom,
                        'papp_a': papp_a_value,
                        'papp_a_mom': papp_mom,
                        'free_beta_hcg': free_beta_hcg_value,
                        'free_beta_hcg_mom': hcg_mom
                    },
                    'risks': risks
                }
                
            else:
                # Иккиламчи скрининг MoM қийматлари
                afp_mom = calculate_mom_value(afp_value, 'AFP', gestational_age, weight, "second")
                total_hcg_mom = calculate_mom_value(total_hcg_value, 'TOTAL_HCG', gestational_age, weight, "second")
                ue3_mom = calculate_mom_value(ue3_value, 'UE3', gestational_age, weight, "second")
                
                marker_moms = {
                    'afp_mom': afp_mom,
                    'total_hcg_mom': total_hcg_mom,
                    'ue3_mom': ue3_mom,
                    'nt_mom': 1.0,  # Суров қилинади
                    'papp_mom': 1.0,  # Суров қилинади
                    'hcg_mom': 1.0   # Суров қилинади
                }
                
                risks = calculate_syndrome_risks(patient_age, marker_moms, "second")
                
                # Бемор маълумотларини тузиш
                patient_data = {
                    'name': patient_name,
                    'age': patient_age,
                    'gestational_age': gestational_age,
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'bmi_category': bmi_category,
                    'screening_type': 'second',
                    'parameters': {
                        'afp': afp_value,
                        'afp_mom': afp_mom,
                        'total_hcg': total_hcg_value,
                        'total_hcg_mom': total_hcg_mom,
                        'ue3': ue3_value,
                        'ue3_mom': ue3_mom
                    },
                    'risks': risks
                }
            
            # Бемор маълумотларини сақлаш
            patient_id = save_patient_record(patient_data)
            st.session_state.current_patient = patient_data
            
            # МУВАФФАҚИЯТЛИ ХАВФ ҲИСОБЛАНДИ
            st.success(f"✅ **{patient_name}** учун генетик хавфлар муваффақиятли ҳисобланди! Пациент ID: `{patient_id}`")
            
            # ==================== БЕМОР МАЪЛУМОТЛАРИ КАРДАСИ ====================
            st.markdown("### 📋 БЕМОР МАЪЛУМОТЛАРИ")
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            with col_p1:
                st.metric("👤 **Бемор**", patient_name)
            
            with col_p2:
                st.metric("🎂 **Ёши**", f"{patient_age} йош")
            
            with col_p3:
                st.metric("🤰 **Хомилалик**", f"{gestational_age} ҳафта")
            
            with col_p4:
                st.metric("📊 **BMI**", f"{bmi:.1f}", bmi_category)
            
            st.markdown("---")
            
            # ==================== ГЕНЕТИК СИНДРОМЛАР ХАВФЛАРИ ====================
            st.markdown("### 🧬 ГЕНЕТИК СИНДРОМЛАР ХАВФЛАРИ")
            
            # Ҳар бир синдром учун карта яратиш
            for syndrome_key in ['downs', 'edwards', 'patau', 'turner', 'ntd']:
                syndrome_info = SYNDROME_DESCRIPTIONS[syndrome_key]
                risk_value = risks.get(syndrome_key, 0)
                risk_display = format_risk_display(risk_value)
                category, risk_class, _ = get_risk_category(risk_value)
                
                css_class = f"{syndrome_key}-card".replace('_', '-')
                
                with st.container():
                    st.markdown(f'<div class="syndrome-card {css_class}">', unsafe_allow_html=True)
                    
                    col_s1, col_s2, col_s3 = st.columns([3, 2, 3])
                    
                    with col_s1:
                        st.markdown(f"#### {syndrome_info['icon']} **{syndrome_info['name']}**")
                        st.markdown(f"*({syndrome_info['scientific']})*")
                        st.markdown(f"**Хусусият:** {syndrome_info['description']}")
                    
                    with col_s2:
                        st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
                        st.metric("**Хавф нисбати**", risk_display)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_s3:
                        st.markdown(f"<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown(f'<div class="{risk_class}">{category}</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # ==================== ЁШ ХАВФЛАРИ КАРДАСИ ====================
            if 'age_risk' in risks:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("#### 📊 ЁШ БЎЙИЧА ХАВФ КЎПАЙТИРУВЧИЛАРИ")
                
                age_risks = risks['age_risk']
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                
                with col_a1:
                    st.metric("**Даун синдроми**", f"{age_risks.get('downs', 1.0):.1f}x")
                
                with col_a2:
                    st.metric("**Эдвардс синдроми**", f"{age_risks.get('edwards', 1.0):.1f}x")
                
                with col_a3:
                    st.metric("**Патау синдроми**", f"{age_risks.get('patau', 1.0):.1f}x")
                
                with col_a4:
                    st.metric("**Тернер синдроми**", f"{age_risks.get('turner', 1.0):.1f}x")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ==================== ГРАФИК ТАҲЛИЛ ====================
            st.markdown("### 📈 ХАВФ ТАҲЛИЛИ")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                # Бар график
                syndromes = [SYNDROME_DESCRIPTIONS[key]['name'] for key in ['downs', 'edwards', 'patau', 'turner', 'ntd']]
                risk_values = [risks[key] for key in ['downs', 'edwards', 'patau', 'turner', 'ntd']]
                
                # Хавф нисбатлари (1:N)
                risk_ratios = [1/val if val > 0 else 10000 for val in risk_values]
                
                fig_bar = px.bar(
                    x=syndromes,
                    y=risk_ratios,
                    title="Генетик синдромлар хавфлари (1:N нисбат)",
                    labels={'x': 'Синдром', 'y': 'Хавф нисбати (1:N)'},
                    color=syndromes,
                    color_discrete_sequence=['#ff6b6b', '#ff9800', '#ff5722', '#9c27b0', '#4caf50']
                )
                
                fig_bar.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title="Хавф нисбати (қанчада 1 та)",
                    xaxis_title=""
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_g2:
                # Ёш хавф графиги
                ages = list(AGE_RISK_MULTIPLIERS.keys())
                age_values = list(range(20, 46, 5))
                
                fig_age = go.Figure()
                
                # Ҳар бир синдром учун чизиқ
                syndromes_plot = ['downs', 'edwards', 'patau']
                colors = ['#ff6b6b', '#ff9800', '#ff5722']
                names = ['Даун', 'Эдвардс', 'Патау']
                
                for idx, syndrome in enumerate(syndromes_plot):
                    multipliers = [get_age_risk_multiplier(age, syndrome) for age in age_values]
                    
                    fig_age.add_trace(go.Scatter(
                        x=age_values,
                        y=multipliers,
                        mode='lines+markers',
                        name=names[idx],
                        line=dict(color=colors[idx], width=3),
                        marker=dict(size=8)
                    ))
                
                fig_age.update_layout(
                    title="Ёш бўйича генетик синдромлар хавфи",
                    xaxis_title="Онанинг ёши",
                    yaxis_title="Хавф кўпайтирувчиси",
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Жорий ёшни белгилаш
                fig_age.add_vline(
                    x=patient_age,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Жорий ёш: {patient_age}",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig_age, use_container_width=True)
            
            # ==================== МАРКЕРЛАР ТАҲЛИЛИ ====================
            st.markdown("### 🔬 МАРКЕРЛАР ТАҲЛИЛИ")
            
            if st.session_state.screening_type == "first":
                markers_data = [
                    ("NT", nt_value, nt_mom, "мм", 2.5, ">"),
                    ("PAPP-A", papp_a_value, papp_mom, "U/L", 0.4, "<"),
                    ("Free β-hCG", free_beta_hcg_value, hcg_mom, "ng/ml", 2.0, ">")
                ]
            else:
                markers_data = [
                    ("AFP", afp_value, afp_mom, "ng/ml", 2.0, ">"),
                    ("Total hCG", total_hcg_value, total_hcg_mom, "IU/L", 2.0, ">"),
                    ("uE3", ue3_value, ue3_mom, "nmol/L", 0.5, "<")
                ]
            
            cols_markers = st.columns(3)
            
            for idx, (name, value, mom, unit, threshold, direction) in enumerate(markers_data):
                with cols_markers[idx]:
                    st.markdown(f"**{name}**")
                    st.metric("Қиймат", f"{value} {unit}")
                    st.metric("MoM", f"{mom:.2f}")
                    
                    # Нормал ёки ненормалликни кўрсатиш
                    if direction == ">" and value > threshold:
                        st.error(f"⛔ Юқори (норма: <{threshold} {unit})")
                    elif direction == "<" and value < threshold:
                        st.error(f"⛔ Паст (норма: >{threshold} {unit})")
                    else:
                        st.success("✅ Нормал диапазонда")
            
            # ==================== ТАВСИЯЛАР ====================
            st.markdown("### 💡 ТИББИЙ ТАВСИЯЛАР")
            
            # Энг юқори хавфли синдромни аниқлаш
            max_risk = 0
            max_syndrome = ""
            
            for syndrome_key in ['downs', 'edwards', 'patau', 'turner', 'ntd']:
                risk_val = risks.get(syndrome_key, 0)
                if risk_val > max_risk:
                    max_risk = risk_val
                    max_syndrome = SYNDROME_DESCRIPTIONS[syndrome_key]['name']
            
            max_risk_display = format_risk_display(max_risk)
            
            with st.expander("#### 🏥 Хавф даражасига кўра тавсиялар", expanded=True):
                st.markdown(f"**Энг юқори хавф:** {max_syndrome} ({max_risk_display})")
                
                if max_risk > 0.05:  # 1:20 дан юқори
                    st.markdown("""
                    ### 🔴 **ШАФФОФ ЧОРАЛАР ТАВСИЯ ЕТИЛАДИ:**
                    
                    **ДАРОР ЧОРАЛАРИ (24 соат ичида):**
                    1. **Дарҳол генетик машварат** - мутахассис генетикга мурожаат
                    2. **NIPT тести** - но-инвазив пренатал тест (қон тести)
                    3. **Инвазив диагностика** - амниоцентез ёки хорион биопсияси
                    4. **Фетал эхокардиография** - юракни детал текшириш
                    5. **Ҳар ҳафта ультратовуш** - доимий мониторинг
                    
                    **ҚОШИМЧА ТАДҚИҚОТЛАР:**
                    - Кариотип таҳлили
                    - Микрочип таҳлили (CMA)
                    - WES тести (Whole Exome Sequencing)
                    """)
                    
                elif max_risk > 0.01:  # 1:100
                    st.markdown("""
                    ### 🟠 **ОЧИҚ ЧОРАЛАР ТАВСИЯ ЕТИЛАДИ:**
                    
                    **ТЕЗ ТЕКШИРИШ (72 соат ичида):**
                    1. **Генетик машварат** - детал маълумот ва ёрим
                    2. **Деталли ультратовуш** - 2-даражали скрининг
                    3. **Қўшимча тестлар** - НIPT ёки квад тест
                    4. **Мунтазам мониторинг** - ҳар 2 ҳафтада назорат
                    
                    **МОДДА АЛМАШИНУВИ:**
                    - Фолат кислотаси (4 мг/кун)
                    - Витамин B комплекс
                    - Йод препаратлари
                    """)
                    
                elif max_risk > 0.001:  # 1:1000
                    st.markdown("""
                    ### 🟡 **НАЗОРАТ ЧОРАЛАРИ:**
                    
                    **МУНТАЗАМ КУЗАТУВ:**
                    1. **Стандарт мониторинг** - регламент тартибида ультратовуш
                    2. **Генетик машварат** - ихтиёрий, агар керак бўлса
                    3. **Парвардалик кўрсатмалари** - соглом турмуш тарзи
                    4. **Ҳар 4-6 ҳафтада** - назорат ўтказиш
                    
                    **ПРОФИЛАКТИКА:**
                    - Муқим парвардалик
                    - Стрессдан сақланиш
                    - Муносиб озиқ-овқат
                    """)
                    
                else:  # 1:1000 дан паст
                    st.markdown("""
                    ### 🟢 **НОРМАЛ ПАРВАРДАЛИК:**
                    
                    **СТАНДАРТ ДАВОЛ ДАСТУРИ:**
                    1. **Регламент скрининг** - плантирилган тартибда текшириш
                    2. **Мунтазам ультратовуш** - тайинланган муддатларда
                    3. **Соглом турмуш тарзи** - тавсия этилган озиқ-овқат
                    4. **Даво-профилактика** - витамин ва минераллар
                    
                    **МАШВАРАТ:**
                    - Ҳар қандай шубҳа бўлса, шифокорга мурожаат
                    - Қўшимча маълумот учун генетик машварат
                    """)
            
            # ==================== БЕМОР ТАРИХИ ====================
            patient_history = get_patient_summary()
            if patient_history:
                with st.expander("#### 📊 ОХИРГИ БЕМОРЛАР ТАРИХИ", expanded=False):
                    for patient in patient_history:
                        with st.container():
                            col_h1, col_h2, col_h3, col_h4 = st.columns([3, 2, 2, 3])
                            
                            with col_h1:
                                st.markdown(f"**{patient['name']}** ({patient['age']}й)")
                            
                            with col_h2:
                                st.caption(f"Ҳафта: {patient['gestational_age']}")
                            
                            with col_h3:
                                risk_val = patient.get('downs_risk', 0)
                                if risk_val > 0:
                                    st.caption(f"Даун: 1:{int(1/risk_val)}")
                            
                            with col_h4:
                                st.caption(patient.get('timestamp', ''))
                        
                        st.divider()
        
        except Exception as e:
            st.error(f"❌ **ХАТОЛИК:** Ҳисоблаш жараёнида хатолик юз берди: {str(e)}")
            st.info("Илтимос, барча маълумотларни қайта текшириб, қайта уриниб кўринг.")

else:
    # ==================== КИРИШ САҲИФАСИ ====================
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%); color: white; padding: 40px; border-radius: 20px; margin: 20px 0;">
        <h2 style="text-align: center; margin-bottom: 20px;">🧬 ГЕНЕТИК СИНДРОМЛАР ХАВФ БАХОЛАШ ДАСТУРИГА ХУШ КЕЛИБСИЗ!</h2>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px;">
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                <h3>👶 Даун синдроми</h3>
                <p><strong>Трисомия 21</strong> - интеллектуал нотўликлик, юрак аномалиялари, мускул гипотонияси</p>
                <p><em>Хавф омиллари:</em> Ҳар иккала ота-онада ёш, оилда борилиги</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                <h3>⚠️ Эдвардс синдроми</h3>
                <p><strong>Трисомия 18</strong> - оғир кўп орган зарарланиши, йўл-йўлақа аномалиялари</p>
                <p><em>Хавф омиллари:</em> Онанинг ёши, қийин вазн орттириш</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                <h3>🔬 Патау синдроми</h3>
                <p><strong>Трисомия 13</strong> - неврологик аномалиялар, кўз ва юз аномалиялари</p>
                <p><em>Хавф омиллари:</em> Ота-она ёши, радиацияга мулоқот</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                <h3>🧬 Тернер синдроми</h3>
                <p><strong>45,X</strong> - бўй пастлиги, жинсий руксатсизлик, юрак аномалиялари</p>
                <p><em>Хавф омиллари:</em> Отанинг ёши, модда алмашинуви</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                <h3>📏 Нейротубуляр дефект</h3>
                <p><strong>НТД</strong> - спина бифида, анэнцефалия, менингоцеле</p>
                <p><em>Хавф омиллари:</em> Фолат етишмовчилиги, диабет, ожирение</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;">
                <h3>🎂 Ёш хавфи</h3>
                <p><strong>35+ ёш</strong> - генетик аномалиялар хавфи каттарок</p>
                <p><em>Муҳим:</em> 35 ёшдан сўнг хавф асосий омил ҳисобланади</p>
            </div>
            
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: rgba(255,255,255,0.15); border-radius: 10px;">
            <h3>📋 ДАСТУРНИ ИШЛАТИШ УЧУН ҚАДАМЛАР:</h3>
            <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-top: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">1️⃣</div>
                    <p>Чеп томондаги панелда барча маълумотларни тўлдиринг</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">2️⃣</div>
                    <p>Скрининг турини танланг (биринчи ёки иккиламчи)</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">3️⃣</div>
                    <p>«ГЕНЕТИК ХАВФЛАРНИ ҲИСОБЛАШ» тугмасини босинг</p>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 30px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <p style="text-align: center; font-style: italic;">
                <strong>DELFIA Revvity</strong> реагентлари асосида ишлаб чиқилган. 
                Биринчи ва иккиламчи триместр скрининглари учун мослаштирилган.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== ФУТЕР ====================
st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p style="font-size: 1.1rem; font-weight: bold; color: #0d47a1;">
        © 2024 ГЕНЕТИК СИНДРОМЛАР ХАВФ БАХОЛАШ ДАСТУРИ | DELFIA Revvity асосида
    </p>
    <div class="warning-box">
        <p style="font-size: 0.9rem; font-weight: bold;">
            ⚕️ <strong>ТИББИЙ ОГОҲЛАНТИРИШ:</strong> Бу дастур фақат ёрдамчи восита сифатида ишлатилади. 
            Ҳеч қандай ҳолда тиббий қарор қабул қилиш учун ёлғиз асос бўлиб хизмат қилмайди. 
            Ҳар қандай тиббий қарор қабул қилишдан олдин мутахассис шифокорга мурожаат қилинг.
        </p>
        <p style="font-size: 0.8rem; margin-top: 10px;">
            Дастур базасида илмий адабиётлар, клиник кўрсатмалар ва DELFIA Revvity нормалари асосида ишлаб чиқилган.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== ЯШИРИН ТЕКШИРИШ ====================
if st.sidebar.checkbox("👨‍💻 Дастурчи режими", help="Техник маълумотлар"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Техник маълумотлар")
    
    st.sidebar.metric("Streamlit версияси", st.__version__)
    st.sidebar.metric("Pandas версияси", pd.__version__)
    st.sidebar.metric("NumPy версияси", np.__version__)
    st.sidebar.metric("Plotly версияси", px.__version__)
    
    if 'current_patient' in st.session_state and st.session_state.current_patient:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Охирги ҳисоблаш")
        st.sidebar.json(st.session_state.current_patient, expanded=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### Сессия маълумотлари")
    st.sidebar.metric("Беморлар сони", len(st.session_state.patient_history))
    st.sidebar.metric("Скрининг тури", st.session_state.screening_type)

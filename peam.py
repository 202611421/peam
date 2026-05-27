import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import koreanize_matplotlib  # ✨ 이 한 줄만 추가하면 자동으로 한글 폰트가 세팅됩니다.
import kagglehub
import os

# 마이너스 기호 깨짐 방지 (이것만 남겨둡니다)
plt.rcParams['axes.unicode_minus'] = False

# 1. 스트림릿 페이지 레이아웃 설정
st.set_page_config(page_title="폐암 환자 군집 분석", layout="wide")

st.title("🩺 폐암 환자 분석 및 군집 예측 시스템")
st.markdown("사용자의 **나이, 흡연량, 음주량**을 입력하면 대시보드 그래프가 실시간으로 반응하여 환자의 위치를 추적합니다.")

# 2. 피클 모델 및 시각화용 원본 데이터 로드 (캐싱 처리)
@st.cache_resource
def load_resources():
    # 모델 로드
    with open("peam.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    # 시각화 배경용 원본 데이터 다운로드 및 가공 (코랩 코드 반영)
    path = kagglehub.dataset_download("yusufdede/lung-cancer-dataset")
    raw_df = pd.read_csv(os.path.join(path, 'lung_cancer_examples.csv'))
    raw_df = raw_df.rename(columns={
        'Age': '나이', 'Smokes': '흡연량', 'Alkhol': '음주량'
    })
    
    # 로드한 모델로 기존 데이터 군집 라벨링 사전 동기화
    X_raw = raw_df[['나이', '흡연량', '음주량']]
    X_raw_scaled = model_data["scaler"].transform(X_raw)
    raw_df['cluster'] = model_data["model"].predict(X_raw_scaled)
    
    return model_data["scaler"], model_data["model"], raw_df

try:
    scaler, model, df = load_resources()
except FileNotFoundError:
    st.error("⚠️ 'peam.pkl' 파일을 찾을 수 없습니다. peam.py와 같은 폴더에 위치해 있는지 확인해 주세요.")
    st.stop()

# 3. 사이드바 - 실시간 입력 슬라이더 컨트롤러
st.sidebar.header("📋 실시간 환자 데이터 조정")
smokes = st.sidebar.slider("흡연량 입력 (Smokes):", min_value=0, max_value=40, value=35)
alkhol = st.sidebar.slider("음주량 입력 (Alkhol):", min_value=0, max_value=15, value=5)
age = st.sidebar.slider("나이 입력 (Age):", min_value=10, max_value=90, value=17)

# 4. 입력 즉시 실시간 데이터 연산 가동
new_patient = pd.DataFrame([[age, smokes, alkhol]], columns=['나이', '흡연량', '음주량'])
new_patient_scaled = scaler.transform(new_patient)
pred_cluster = model.predict(new_patient_scaled)[0]

# 5. 화면 분할 대시보드 레이아웃
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🔍 실시간 군집 분석")
    
    # 동적 스태터스 박스 구현
    st.metric(label="예측된 소속 군집 번호", value=f"{pred_cluster}번 군집")
    
    st.markdown("---")
    st.markdown("### 💡 실시간 상태 프로파일링 결과")
    
    # 지상님 크로스탭 분석 데이터 결과 기반 프로파일링 매핑
    if pred_cluster == 3:
        st.warning("🚨 **군집 3 (고위험 유해 습관군):** 흡연량과 음주 자극이 결합되어 발병률 지표가 급증하는 위험 패턴 영역입니다. 즉각적인 정기 검진 및 정밀 진단 수립을 권장합니다.")
    elif pred_cluster == 0:
        st.info("✅ **군집 0 (청년층 안전군):** 전반적인 유해 습관 빈도 지표가 낮고 건강한 상태를 유지 중인 베이스라인 그룹입니다.")
    elif pred_cluster == 1:
        st.error("⚠️ **군집 1 (고연령 만성 위험군):** 고연령 및 오랜 기간 노출된 생활 습관 축적으로 인해 실제 폐암 발병 여부(Result=1)가 집중적으로 발현되는 최우선 관리 대상군입니다.")
    else:
        st.info("🟡 **군집 2 (중년층 잠재 관리군):** 중간 단계의 습관성 노출 지표를 보이며 생활 패턴 개선 시 예방 효율이 가장 극대화되는 타깃 그룹입니다.")

with col2:
    st.subheader("📊 환자 위치 공간 시각화 (흡연량 vs 음주량)")
    
    # 실시간 다이내믹 차트 빌드
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    # 1) 배경: 기존 대조 환자 그룹 점 찍기 (cmap 컬러로 군집 구분)
    scatter = ax.scatter(df['흡연량'], df['음주량'], c=df['cluster'], cmap='Set2', alpha=0.6, s=60, label='기존 대조 환자군')
    
    # 2) 핵심: 사용자가 사이드바를 움직일 때마다 실시간으로 쫓아다니는 실시간 마커 구동
    ax.scatter(smokes, alkhol, c='black', s=450, marker='X', edgecolors='white', linewidths=2, label='현재 입력 환자 위치', zorder=10)
    
    # 그래프 스케일링 가독성 고정 및 라벨링
    ax.set_xlim(-2, 40)
    ax.set_ylim(-1, 12)
    ax.set_xlabel('흡연량 (Smokes)', fontsize=11)
    ax.set_ylabel('음주량 (Alkhol)', fontsize=11)
    ax.set_title('환자 행동 데이터 공간 내 실시간 매핑 위치', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    st.pyplot(fig)
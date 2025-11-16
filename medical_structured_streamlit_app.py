"""
Streamlit app to calculate the value of structured medical data (e.g., lab tests)

This application mirrors the functionality of the original Excel‐based "정형 데이터 가치 평가
계산기".  Users can select a medical procedure (행위), choose the size of the medical facility
and organisation type, specify the data management type (정형 vs various 비정형 types), and
indicate the scarcity category of the associated disease.  The app then calculates
step‑wise costs and final values using formulas analogous to those used in the
unstructured X‑ray calculator, but with constants extracted from the Excel file rather
than an HTML file.

To use this app, ensure that the Excel file ``Medical Data Value calculator_정형_alpha_1.1_version - 복사본.xlsx``
resides in the same directory as this script or update ``EXCEL_FILE_PATH`` accordingly.

The app is intended for educational and informational purposes only.  Real‑world
pricing and weighting may differ.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to the Excel file containing the structured data value calculator.
# Modify this if you place the Excel file elsewhere.
EXCEL_FILE_PATH: str = "Medical Data Value calculator_정형_alpha_1.1_version - 복사본.xlsx"


@st.cache_data
def load_dictionaries_from_excel(xl_path: str) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]
]:
    """Parse the Excel file and extract cost/weight dictionaries.

    The Excel workbook contains many interleaved tables.  This function reads the
    relevant rows and columns to build dictionaries that mirror those in the
    unstructured X‑ray app: procedure costs (act_to_cost), facility size weights
    (scale_to_weight), institution consultation costs (inst_to_cost), data type
    weights (data_type_to_weight) and disease scarcity weights (disease_to_scarcity).

    Parameters
    ----------
    xl_path : str
        Path to the Excel workbook.

    Returns
    -------
    tuple
        Five dictionaries in the order: act_to_cost, scale_to_weight,
        inst_to_cost, data_type_to_weight, disease_to_scarcity.

    Raises
    ------
    FileNotFoundError
        If the Excel file cannot be found at the provided path.
    ValueError
        If required columns are missing or parsing fails.
    """
    excel_file = Path(xl_path)
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    # Read the sheet with the data.  Header row is at index 3 (zero‑based), so
    # header=3 ensures proper column names.
    df = pd.read_excel(excel_file, sheet_name=0, header=3)

    # Build act_to_cost: map each unique procedure name (행위) to its cost (단가.1).
    act_to_cost: Dict[str, float] = {}
    if "행위" not in df.columns or "단가.1" not in df.columns:
        raise ValueError("Expected columns '행위' and '단가.1' not found in Excel file")
    act_df = df[["행위", "단가.1"]].dropna()
    for act, cost in act_df.drop_duplicates(subset="행위").itertuples(index=False):
        try:
            act_to_cost[str(act)] = float(cost)
        except (ValueError, TypeError):
            # Skip rows with non‑numeric costs
            continue

    # Build scale_to_weight: facility size (규모) to weight.  The weights
    # for the facility size categories are stored in the '가중치.1' column (not '가중치'),
    # so use that column for extraction.  This ensures that all four size
    # categories have numeric weights.
    scale_to_weight: Dict[str, float] = {}
    if "규모" not in df.columns or "가중치.1" not in df.columns:
        raise ValueError("Expected columns '규모' and '가중치.1' not found in Excel file")
    size_df = df[["규모", "가중치.1"]].dropna()
    for size, weight in size_df.drop_duplicates(subset="규모").itertuples(index=False):
        try:
            weight_val = float(weight)
        except (ValueError, TypeError):
            continue
        scale_to_weight[str(size)] = weight_val

    # Define known data type categories to separate them from facility sizes
    data_type_categories = [
        "정형",
        "비정형_x-ray",
        "비정형_시그널",
        "비정형_음성",
        "비정형_영상",
        "비정형_유전체",
    ]
    # Remove data type categories from scale_to_weight if they were inadvertently included
    for key in list(scale_to_weight.keys()):
        if key in data_type_categories:
            scale_to_weight.pop(key)

    # Build inst_to_cost: institution type (규모.1) to consultation cost (진찰료_판정료).
    inst_to_cost: Dict[str, float] = {}
    if "규모.1" not in df.columns or "진찰료_판정료" not in df.columns:
        raise ValueError(
            "Expected columns '규모.1' and '진찰료_판정료' not found in Excel file"
        )
    inst_df = df[["규모.1", "진찰료_판정료"]].dropna()
    for inst, cost in inst_df.drop_duplicates(subset="규모.1").itertuples(index=False):
        try:
            inst_to_cost[str(inst)] = float(cost)
        except (ValueError, TypeError):
            continue

    # Build data_type_to_weight: data management type to weight.
    # These appear under the '규모' column after a header row labelled '데이터 관리 유형'.
    data_type_to_weight: Dict[str, float] = {}
    # Identify rows where the '규모' column matches known data types and the
    # weight is in '가중치.1'.  We skip the header row itself (값 '가중치').
    if "규모" not in df.columns or "가중치.1" not in df.columns:
        raise ValueError("Expected columns '규모' and '가중치.1' for data type weights")
    dt_df = df[["규모", "가중치.1"]].dropna()
    for dtype, weight in dt_df.itertuples(index=False):
        try:
            weight_val = float(weight)
        except (ValueError, TypeError):
            continue
        data_type_to_weight[str(dtype)] = weight_val
    # Exclude any sizes that also appear in scale_to_weight to avoid confusion
    for key in list(data_type_to_weight.keys()):
        if key in scale_to_weight:
            data_type_to_weight.pop(key)

    # Build disease_to_weight: map each disease name to its scarcity weight.
    disease_to_weight: Dict[str, float] = {}
    # Ensure necessary columns are present
    for col in ["질병", "희소가중치"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in Excel file")
    disease_df = df[["질병", "희소가중치"]].dropna().drop_duplicates(subset="질병")
    for disease, weight in disease_df.itertuples(index=False):
        try:
            disease_to_weight[str(disease)] = float(weight)
        except (ValueError, TypeError):
            continue

    return act_to_cost, scale_to_weight, inst_to_cost, data_type_to_weight, disease_to_weight


def calculate_values(
    act: str,
    size: str,
    institution: str,
    disease: str,
    data_type: str,
    quality: str,
    act_to_cost: Dict[str, float],
    scale_to_weight: Dict[str, float],
    inst_to_cost: Dict[str, float],
    data_type_to_weight: Dict[str, float],
    disease_to_weight: Dict[str, float],
) -> Dict[str, float]:
    """Compute step‑wise costs and values for structured medical data.

    The computation mirrors the logic used in the unstructured X‑ray calculator.
    We apply facility size weights, add consultation costs, apply storage and
    data type weights, optionally zero out cost on quality failure, and then
    incorporate scarcity, effectiveness, distribution and tax weights.

    Parameters
    ----------
    act : str
        Selected medical procedure.
    size : str
        Selected facility size (규모) affecting the size weight.
    institution : str
        Selected institution type (규모.1) affecting consultation cost.
    disease_cat : str
        Disease rarity category (G1–G7) used to apply scarcity weight.
    data_type : str
        Data management type affecting distribution weight.
    quality : str
        Quality judgment ('Pass' or 'Fail'); failing sets cost to zero after
        applying storage weight.
    act_to_cost : Dict[str, float]
        Mapping from act names to base costs.
    scale_to_weight : Dict[str, float]
        Mapping from facility sizes to weights.
    inst_to_cost : Dict[str, float]
        Mapping from institution types to consultation costs.
    data_type_to_weight : Dict[str, float]
        Mapping from data management types to distribution weights.
    disease_to_scarcity : Dict[str, float]
        Mapping from disease categories to scarcity weights (negative values).

    Returns
    -------
    Dict[str, float]
        A dictionary of descriptive names to the computed values at each step.
    """
    # Step 1: base cost for the selected procedure
    base_cost = float(act_to_cost.get(act, 0.0))
    # Step 2: apply facility size weight
    size_weight = float(scale_to_weight.get(size, 0.0))
    step2_cost = base_cost * (1.0 + size_weight)
    # Step 3: add consultation cost (institution cost)
    consult_cost = float(inst_to_cost.get(institution, 0.0))
    # Storage weight (원본 데이터 저장 및 관리비).  Use 0.1 as per Excel constant.
    storage_weight = 0.1
    step3_cost = step2_cost + consult_cost * (1.0 + storage_weight)
    # Step 4: apply quality judgment.  Pass retains cost, Fail zeroes it out.
    step4_cost = step3_cost if quality == "Pass" else 0.0
    # Step 5: apply disease scarcity weight
    disease_weight = float(disease_to_weight.get(disease, 0.0))
    step5_cost = step4_cost * (1.0 + disease_weight)
    # Step 6: apply effectiveness weight (use constant 0.2)
    effectiveness_weight = 0.2
    step6_cost = step5_cost * (1.0 + effectiveness_weight)
    # Step 7: apply distribution weight from data type
    distribution_weight = float(data_type_to_weight.get(data_type, 0.0))
    step7_cost = step6_cost * (1.0 + distribution_weight)
    # Step 8: apply tax weight (법정 세금) constant 0.1
    tax_weight = 0.1
    final_value = step7_cost * (1.0 + tax_weight)
    # Additional outputs analogous to unstructured app
    distribution_value_without_disease = final_value * 0.6
    net_value_with_disease = step6_cost
    net_value_without_disease = net_value_with_disease * 0.6

    return {
        "데이터 생성 비용 (Step 1)": base_cost,
        "의료기관 규모 가중치 적용 후 비용 (Step 2)": step2_cost,
        "진찰료 (Step 3)": consult_cost,
        "저장 및 관리비 가중치 적용 후 비용 (Step 4)": step3_cost,
        "품질 평가 후 비용 (Step 5)": step4_cost,
        "질병 희소가중치 적용 후 비용 (Step 6)": step5_cost,
        "효과성 가중치 적용 후 비용 (Step 7)": step6_cost,
        "유통수수료 가중치 적용 후 비용 (Step 8)": step7_cost,
        "법정 세금 적용 후 최종 유통가치 (Step 9)": final_value,
        "최종 유통가치 (질병 미포함)": distribution_value_without_disease,
        "최종 순가치 (질병 포함)": net_value_with_disease,
        "최종 순가치 (질병 미포함)": net_value_without_disease,
    }


def main() -> None:
    """Render the Streamlit UI and handle user interaction."""
    # Attempt to load dictionaries.  If loading fails, show a friendly error.
    try:
        (
            act_to_cost,
            scale_to_weight,
            inst_to_cost,
            data_type_to_weight,
            disease_to_weight,
        ) = load_dictionaries_from_excel(EXCEL_FILE_PATH)
    except Exception as exc:
        st.error(f"데이터 사전을 로드하는 중 오류가 발생했습니다: {exc}")
        st.stop()

    # Main page title and description styled similarly to the X‑ray valuation tool
    st.title("정형 의료 데이터 ‒ 진단검사 데이터 가치 계산기")
    st.write(
        "아래의 옵션을 선택한 후 계산하기 버튼을 눌러 단계별 비용과 가치를 확인하세요."
    )

    # Prepare options for selection boxes.  Sort keys for consistency.
    act_options = sorted(act_to_cost.keys())
    # Limit facility size options to the four specific categories requested
    allowed_sizes = ["의원_소상공인", "병원_소기업", "종합병원_중기업", "상급종합병원_대기업"]
    # For consistency with the X‑ray tool and to satisfy the user's requirement,
    # always offer exactly the four facility size categories, regardless of
    # whether they are present in the loaded dictionary.  The weights will
    # still be looked up in ``scale_to_weight`` when computing results, defaulting
    # to 0.0 if a size is missing.
    size_options = allowed_sizes
    inst_options = list(inst_to_cost.keys())
    disease_options = list(disease_to_weight.keys())
    # Fix data type to '정형' only
    # Fix data type to '정형' only regardless of dictionary contents
    data_type_options = ["정형"]

    with st.form(key="calculator_form"):
        act = st.selectbox("의료행위 (정형) 선택", act_options, key="act")
        size = st.selectbox("의료기관 규모 선택", size_options, key="size")
        institution = st.selectbox("진찰료 기준 의료기관 유형 선택", inst_options, key="institution")
        disease = st.selectbox("질병 선택", disease_options, key="disease")
        data_type = st.selectbox("데이터 관리 유형 선택", data_type_options, key="data_type")
        quality = st.selectbox("품질 판정", ["Pass", "Fail"], key="quality")
        submit = st.form_submit_button("계산하기")

    if submit:
        results = calculate_values(
            act,
            size,
            institution,
            disease,
            data_type,
            quality,
            act_to_cost,
            scale_to_weight,
            inst_to_cost,
            data_type_to_weight,
            disease_to_weight,
        )
        result_df = pd.DataFrame(
            {
                "단계": list(results.keys()),
                "값 (원)": [f"{value:,.2f}" for value in results.values()],
            }
        )
        # Ensure the index starts at 0 for visual consistency with the X‑ray tool
        result_df.index = range(len(result_df))
        st.subheader("계산 결과")
        st.table(result_df)
        # Optional footer similar to the X‑ray tool
        st.caption("Created by curein")


if __name__ == "__main__":
    main()
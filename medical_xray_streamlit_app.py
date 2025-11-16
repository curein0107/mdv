"""
Streamlit app to calculate the value of unstructured medical image data (X‑ray)

This application reproduces the functionality of the original HTML file
``Medical Data Value calculator_비정형(x-ray)_alpha_1.1_version (2024).html``
as a Streamlit web app.  Users select parameters such as the medical
procedure (act), facility size, institution type, disease, data type and
quality.  The app then computes the step‑wise cost and value according to
the same formulas used in the original JavaScript implementation and
displays the results in a table.

The cost/weight dictionaries are not hard coded here.  Instead, they are
extracted at runtime from the original HTML file using simple regular
expressions and JSON parsing.  By doing so, the app stays up to date with
the source data and avoids duplicating several hundred lines of constant
definitions.

To use this app, make sure the HTML file is located in the same
directory as this Python script or specify its path via the
``HTML_FILE_PATH`` constant below.  The app will raise a clear error if
the file cannot be found or parsed.

This app is intended for educational and informational purposes only.
Actual pricing, weights and formulas may vary depending on the
healthcare environment.
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Set this to the relative or absolute path of the original HTML file
# containing the constant definitions.  By default it assumes the file is in
# the same directory as this script.
HTML_FILE_PATH: str = "Medical Data Value calculator_비정형(x-ray)_alpha_1.1_version (2024).html"


@st.cache_data
def load_dictionaries(html_path: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Load the cost/weight dictionaries from the HTML file.

    Parameters
    ----------
    html_path : str
        Path to the HTML file containing the JavaScript constant definitions.

    Returns
    -------
    tuple
        A tuple of five dictionaries: actToCost, scaleToWeight,
        instToCost, dataTypeToWeight and diseaseToScarcity.

    Raises
    ------
    FileNotFoundError
        If the specified HTML file does not exist.
    ValueError
        If any of the constant definitions cannot be extracted or parsed.
    """
    html_file = Path(html_path)
    if not html_file.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file}")

    content = html_file.read_text(encoding="utf-8")
    # Helper to extract a JSON object from a const definition
    def extract_const(name: str) -> Dict[str, float]:
        pattern = rf"const\s+{name}\s*=\s*({{.*?}});"
        match = re.search(pattern, content, re.S)
        if not match:
            raise ValueError(f"Could not find definition for {name} in HTML file")
        obj_str = match.group(1)
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON for {name}: {exc}")

    act_to_cost = extract_const("actToCost")
    scale_to_weight = extract_const("scaleToWeight")
    inst_to_cost = extract_const("instToCost")
    data_type_to_weight = extract_const("dataTypeToWeight")
    disease_to_scarcity = extract_const("diseaseToScarcity")
    return act_to_cost, scale_to_weight, inst_to_cost, data_type_to_weight, disease_to_scarcity


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
    disease_to_scarcity: Dict[str, float],
) -> Dict[str, float]:
    """Compute step‑wise costs and values based on the selected parameters.

    Parameters correspond to the selection boxes in the UI.  Dictionaries are
    looked up for base cost, weights and multipliers.  All intermediate
    values are returned as a dictionary keyed by descriptive names.
    """
    base_cost = float(act_to_cost.get(act, 0.0))
    size_weight = float(scale_to_weight.get(size, 0.0))
    step2_cost = base_cost * (1 + size_weight)
    consult_cost = float(inst_to_cost.get(institution, 0.0))
    storage_weight = 0.1
    step4_cost = step2_cost + consult_cost * (1 + storage_weight)
    # Apply quality: if Fail, cost becomes 0 after step4
    step5_cost = step4_cost if quality == "Pass" else 0.0
    scarcity_weight = float(disease_to_scarcity.get(disease, 0.0))
    step6_cost = step5_cost * (1 + scarcity_weight)
    effectiveness_weight = 0.2
    step7_cost = step6_cost * (1 + effectiveness_weight)
    distribution_weight = float(data_type_to_weight.get(data_type, 0.0))
    step8_cost = step7_cost * (1 + distribution_weight)
    tax_weight = 0.1
    final_distribution_value = step8_cost * (1 + tax_weight)
    distribution_value_without_disease = final_distribution_value * 0.6
    net_value_with_disease = step7_cost
    net_value_without_disease = net_value_with_disease * 0.6

    return {
        "데이터 생성 비용 (Step 1)": base_cost,
        "의료기관 규모 가중치 적용 후 비용 (Step 2)": step2_cost,
        "진찰료 (Step 3)": consult_cost,
        "저장 및 관리비 가중치 적용 후 비용 (Step 4)": step4_cost,
        "품질 평가 후 비용 (Step 5)": step5_cost,
        "희소가중치 적용 후 비용 (Step 6)": step6_cost,
        "효과성 가중치 적용 후 비용 (Step 7)": step7_cost,
        "유통수수료 가중치 적용 후 비용 (Step 8)": step8_cost,
        "법정 세금 적용 후 최종 유통가치 (Step 9)": final_distribution_value,
        "최종 유통가치 (질병 미포함)": distribution_value_without_disease,
        "최종 순가치 (질병 포함)": net_value_with_disease,
        "최종 순가치 (질병 미포함)": net_value_without_disease,
    }


def main() -> None:
    # Load dictionaries with caching.  If loading fails, display an error.
    try:
        act_to_cost, scale_to_weight, inst_to_cost, data_type_to_weight, disease_to_scarcity = load_dictionaries(HTML_FILE_PATH)
    except Exception as e:
        st.error(f"데이터 사전을 로드하는 중 오류가 발생했습니다: {e}")
        st.stop()

    st.title("비정형 의료 이미지 - 방사선단순영상 X-Ray 데이터 가치 계산기 (Streamlit)")
    st.write(
        "이 도구는 기존 HTML 계산기를 Streamlit으로 구현한 것입니다."
        " 아래의 옵션을 선택한 후 `계산하기` 버튼을 눌러 단계별 비용과 가치를 확인하세요."
    )

    # UI: selection boxes.  Sort keys for consistent ordering.
    act_options = sorted(act_to_cost.keys())
    size_options = list(scale_to_weight.keys())
    inst_options = list(inst_to_cost.keys())
    disease_options = sorted(disease_to_scarcity.keys())
    data_type_options = list(data_type_to_weight.keys())

    with st.form(key="calculator_form"):
        act = st.selectbox("의료행위 (X-Ray) 선택", act_options, key="act")
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
            disease_to_scarcity,
        )
        # Convert results to a DataFrame for nicer display
        df = pd.DataFrame(
            {
                "단계": list(results.keys()),
                "값 (원)": [f"{value:,.2f}" for value in results.values()],
            }
        )
        st.subheader("계산 결과")
        st.table(df)


if __name__ == "__main__":
    main()
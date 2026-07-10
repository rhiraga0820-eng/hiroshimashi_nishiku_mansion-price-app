import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor


# =========================================================
# 画面設定
# =========================================================

st.set_page_config(
    page_title="マンション適正価格・将来売却価格予測",
    page_icon="🏢",
    layout="wide",
)


# =========================================================
# ファイルパス
# =========================================================

MODEL_PATH = Path(__file__).with_name(
    "mansion_xgb_model.json"
)

METADATA_PATH = Path(__file__).with_name(
    "mansion_metadata.json"
)


# =========================================================
# モデルとメタデータの読み込み
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "mansion_xgb_model.json が見つかりません。"
            "app.pyと同じフォルダに置いてください。"
        )

    loaded_model = XGBRegressor()
    loaded_model.load_model(MODEL_PATH)

    return loaded_model


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "mansion_metadata.json が見つかりません。"
            "app.pyと同じフォルダに置いてください。"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()

except Exception as error:
    st.error("モデルファイルを読み込めませんでした。")
    st.exception(error)
    st.stop()


# =========================================================
# メタデータ
# =========================================================

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]

numeric_medians = metadata["numeric_medians"]
category_options = metadata["category_options"]

output_feature_names = metadata["output_feature_names"]

onehot_feature_map = metadata["onehot_feature_map"]

metrics = metadata.get("test_metrics", {})


# =========================================================
# 補助関数
# =========================================================

def yen_to_man(value: float) -> float:
    """円を万円へ変換する。"""

    return value / 10_000


def format_man(value: float) -> str:
    """円を万円表記にする。"""

    return f"{yen_to_man(value):,.0f}万円"


def preprocess_input(row: dict) -> np.ndarray:
    """
    アプリの入力値を、XGBoost学習時と同じ列順に変換する。

    scikit-learnのColumnTransformerは使用しない。
    """

    # 学習時の全列を0で作る
    processed = {
        feature_name: 0.0
        for feature_name in output_feature_names
    }

    # -----------------------------------------------------
    # 数値変数
    # -----------------------------------------------------

    for column in numeric_features:

        value = row.get(
            column,
            numeric_medians[column]
        )

        try:
            value = float(value)

        except (TypeError, ValueError):
            value = float(
                numeric_medians[column]
            )

        if np.isnan(value):
            value = float(
                numeric_medians[column]
            )

        output_name = f"num__{column}"

        if output_name in processed:
            processed[output_name] = value

    # -----------------------------------------------------
    # カテゴリ変数
    # -----------------------------------------------------

    for column in categorical_features:

        selected_value = str(
            row.get(column, "不明")
        )

        category_mapping = onehot_feature_map.get(
            column,
            {}
        )

        # 選択カテゴリに対応するダミー変数名を取得
        output_name = category_mapping.get(
            selected_value
        )

        # 基準カテゴリの場合はNoneなので、すべて0のまま
        if (
            output_name is not None
            and output_name in processed
        ):
            processed[output_name] = 1.0

    # 学習時と同じ列順に並べる
    values = [
        processed[column]
        for column in output_feature_names
    ]

    # XGBoostへ渡す2次元配列
    return np.array(
        [values],
        dtype=np.float32
    )


def predict_price(row: dict) -> float:
    """マンション価格を予測する。"""

    processed_input = preprocess_input(row)

    prediction = float(
        model.predict(processed_input)[0]
    )

    return max(prediction, 0.0)


def make_future_row(
    base_row: dict,
    years: int,
    future_policy_rate: float,
    future_cpi: float,
) -> dict:
    """将来時点の入力条件を作る。"""

    future_row = base_row.copy()

    future_row["築年数"] = (
        float(base_row["築年数"])
        + years
    )

    future_row["政策金利"] = (
        future_policy_rate
    )

    future_row["物価指数"] = (
        future_cpi
    )

    return future_row


# =========================================================
# タイトル
# =========================================================

st.title(
    "🏢 マンション適正価格・将来売却価格予測"
)

st.caption(
    "物件条件と金利・物価シナリオから、"
    "現在の適正価格と5年ごとの将来売却価格を予測します。"
)


# =========================================================
# モデルの注意事項
# =========================================================

with st.expander(
    "モデルの前提と注意点",
    expanded=False
):
    st.markdown(
        """
        - 現在価格は、過去のマンション取引データで学習した
          XGBoostによる推定値です。
        - 将来価格は、築年数・政策金利・物価指数を変化させた
          シナリオ予測です。
        - XGBoostは学習範囲外への外挿が得意ではありません。
        - 階数、方角、眺望、管理状態、個別のリフォーム品質など、
          学習データにない条件は反映されません。
        - 鑑定評価や不動産会社の査定価格を保証するものではありません。
        """
    )

    test_r2 = metrics.get("r2")
    test_mae_yen = metrics.get("mae_yen")
    test_rmse_yen = metrics.get("rmse_yen")

    if test_r2 is not None:
        st.write(
            f"テストR²：{float(test_r2):.3f}"
        )

    if test_mae_yen is not None:
        st.write(
            f"テストMAE："
            f"{format_man(float(test_mae_yen))}"
        )

    if test_rmse_yen is not None:
        st.write(
            f"テストRMSE："
            f"{format_man(float(test_rmse_yen))}"
        )


# =========================================================
# 1. 物件情報
# =========================================================

st.subheader("1．物件情報")

left, middle, right = st.columns(3)


with left:

    station = st.selectbox(
        "最寄駅",
        options=category_options[
            "最寄駅：名称"
        ],
    )

    station_minutes = st.number_input(
        "駅徒歩・所要時間（分）",
        min_value=0.0,
        max_value=120.0,
        value=10.0,
        step=1.0,
    )

    area = st.number_input(
        "専有面積（㎡）",
        min_value=10.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
    )


with middle:

    rooms = st.number_input(
        "部屋数",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5,
    )

    age = st.number_input(
        "現在の築年数",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

    renovated = st.selectbox(
        "改装状況",
        options=[0, 1],
        format_func=lambda value:
            "改装済み"
            if value == 1
            else "未改装",
    )


with right:

    structure = st.selectbox(
        "建物構造",
        options=category_options[
            "建物の構造"
        ],
    )

    zoning = st.selectbox(
        "都市計画",
        options=category_options[
            "都市計画"
        ],
    )

    asking_price_man = st.number_input(
        "売出価格（万円）",
        min_value=0.0,
        max_value=100_000.0,
        value=4_500.0,
        step=10.0,
    )


# =========================================================
# 2. 現在の経済条件
# =========================================================

st.subheader("2．現在の経済条件")

macro_left, macro_right = st.columns(2)


with macro_left:

    current_policy_rate = st.number_input(
        "現在の政策金利（%）",
        min_value=-1.0,
        max_value=10.0,
        value=0.5,
        step=0.1,
        format="%.2f",
        help=(
            "モデルを学習したときと"
            "同じ単位で入力してください。"
        ),
    )


with macro_right:

    current_cpi = st.number_input(
        "現在の物価指数",
        min_value=50.0,
        max_value=300.0,
        value=111.9,
        step=0.1,
        format="%.1f",
        help=(
            "モデル学習時と同じ基準年の"
            "物価指数を使用してください。"
        ),
    )


# =========================================================
# 基本入力データ
# =========================================================

base_row = {
    "最寄駅：名称": station,
    "最寄駅：距離（分）": station_minutes,
    "部屋数": rooms,
    "面積（㎡）": area,
    "建物の構造": structure,
    "都市計画": zoning,
    "築年数": age,
    "改装": renovated,
    "政策金利": current_policy_rate,
    "物価指数": current_cpi,
}


# =========================================================
# 現在価格の予測
# =========================================================

try:
    current_prediction = predict_price(
        base_row
    )

except Exception as error:
    st.error(
        "現在価格の予測中にエラーが発生しました。"
    )
    st.exception(error)
    st.stop()


asking_price_yen = (
    asking_price_man
    * 10_000
)

discount_yen = (
    current_prediction
    - asking_price_yen
)

if asking_price_yen > 0:

    discount_rate = (
        discount_yen
        / asking_price_yen
        * 100
    )

else:

    discount_rate = np.nan


# =========================================================
# 3. 現在の適正価格
# =========================================================

st.subheader("3．現在の適正価格")

metric1, metric2, metric3 = st.columns(3)


metric1.metric(
    "AI推定適正価格",
    format_man(current_prediction),
)


metric2.metric(
    "売出価格との差",
    format_man(discount_yen),
    delta=(
        f"{discount_rate:,.1f}%"
        if not np.isnan(discount_rate)
        else None
    ),
)


if np.isnan(discount_rate):

    price_judgement = "判定不可"

elif discount_rate >= 5:

    price_judgement = "割安"

elif discount_rate <= -5:

    price_judgement = "割高"

else:

    price_judgement = "おおむね適正"


metric3.metric(
    "購入価格判定",
    price_judgement,
)


test_mae_yen = metrics.get("mae_yen")

if test_mae_yen is not None:

    mae = float(test_mae_yen)

    lower_price = max(
        current_prediction - mae,
        0
    )

    upper_price = (
        current_prediction + mae
    )

    st.info(
        f"テストMAEを単純な誤差の目安として使うと、"
        f"推定価格帯は約 "
        f"{format_man(lower_price)} ～ "
        f"{format_man(upper_price)} です。"
    )


# =========================================================
# 4. 将来シナリオ
# =========================================================

st.subheader("4．将来売却価格シナリオ")


scenario_left, scenario_middle, scenario_right = (
    st.columns(3)
)


with scenario_left:

    future_horizon = st.slider(
        "何年後まで予測するか",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )


with scenario_middle:

    interest_scenario = st.selectbox(
        "金利シナリオ",
        options=[
            "低金利",
            "標準",
            "高金利",
            "手動入力",
        ],
    )


with scenario_right:

    inflation_scenario = st.selectbox(
        "物価シナリオ",
        options=[
            "物価横ばい",
            "年1%上昇",
            "年2%上昇",
            "年3%上昇",
            "手動入力",
        ],
    )


# =========================================================
# 金利シナリオ
# =========================================================

interest_rate_map = {
    "低金利": 0.5,
    "標準": 1.0,
    "高金利": 2.0,
}


if interest_scenario == "手動入力":

    future_policy_rate = st.number_input(
        "将来の政策金利（%）",
        min_value=-1.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.2f",
    )

else:

    future_policy_rate = (
        interest_rate_map[
            interest_scenario
        ]
    )


# =========================================================
# 物価シナリオ
# =========================================================

inflation_rate_map = {
    "物価横ばい": 0.0,
    "年1%上昇": 0.01,
    "年2%上昇": 0.02,
    "年3%上昇": 0.03,
}


if inflation_scenario == "手動入力":

    inflation_rate_percent = st.number_input(
        "年間物価上昇率（%）",
        min_value=-5.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )

    inflation_rate = (
        inflation_rate_percent
        / 100
    )

else:

    inflation_rate = (
        inflation_rate_map[
            inflation_scenario
        ]
    )


st.write(
    f"選択中の将来政策金利："
    f"**{future_policy_rate:.2f}%**"
)

st.write(
    f"選択中の年間物価変化率："
    f"**{inflation_rate * 100:.1f}%**"
)


# =========================================================
# 将来価格予測
# =========================================================

future_years = list(
    range(
        5,
        future_horizon + 1,
        5
    )
)

future_rows = []


for years in future_years:

    future_cpi = (
        current_cpi
        * ((1 + inflation_rate) ** years)
    )

    future_row = make_future_row(
        base_row=base_row,
        years=years,
        future_policy_rate=future_policy_rate,
        future_cpi=future_cpi,
    )

    future_price = predict_price(
        future_row
    )

    difference_from_current = (
        future_price
        - current_prediction
    )

    change_rate = (
        difference_from_current
        / current_prediction
        * 100
        if current_prediction > 0
        else np.nan
    )

    future_rows.append(
        {
            "時点": f"{years}年後",
            "築年数": future_row[
                "築年数"
            ],
            "政策金利（%）":
                future_policy_rate,
            "物価指数":
                future_cpi,
            "予測売却価格（万円）":
                yen_to_man(future_price),
            "現在価格との差（万円）":
                yen_to_man(
                    difference_from_current
                ),
            "現在価格からの変化率（%）":
                change_rate,
        }
    )


future_df = pd.DataFrame(
    future_rows
)


# =========================================================
# 現在価格をグラフ用に追加
# =========================================================

current_row_for_chart = pd.DataFrame(
    [
        {
            "時点": "現在",
            "築年数": age,
            "政策金利（%）":
                current_policy_rate,
            "物価指数":
                current_cpi,
            "予測売却価格（万円）":
                yen_to_man(
                    current_prediction
                ),
            "現在価格との差（万円）":
                0.0,
            "現在価格からの変化率（%）":
                0.0,
        }
    ]
)

all_prediction_df = pd.concat(
    [
        current_row_for_chart,
        future_df,
    ],
    ignore_index=True,
)


# =========================================================
# 将来予測テーブル
# =========================================================

st.dataframe(
    all_prediction_df.style.format(
        {
            "築年数": "{:,.0f}",
            "政策金利（%）": "{:,.2f}",
            "物価指数": "{:,.1f}",
            "予測売却価格（万円）":
                "{:,.0f}",
            "現在価格との差（万円）":
                "{:+,.0f}",
            "現在価格からの変化率（%）":
                "{:+,.1f}",
        }
    ),
    use_container_width=True,
)


# =========================================================
# 将来価格グラフ
# =========================================================

chart_df = all_prediction_df.set_index(
    "時点"
)[
    ["予測売却価格（万円）"]
]

st.line_chart(
    chart_df,
    use_container_width=True,
)


# =========================================================
# 5. 売却手取り
# =========================================================

st.subheader("5．売却手取りの概算")


sale_cost_rate = st.slider(
    "売却諸費用率（%）",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1,
)


net_rows = []


for _, row in future_df.iterrows():

    gross_price_man = float(
        row["予測売却価格（万円）"]
    )

    sale_cost_man = (
        gross_price_man
        * sale_cost_rate
        / 100
    )

    net_price_man = (
        gross_price_man
        - sale_cost_man
    )

    net_rows.append(
        {
            "時点": row["時点"],
            "予測売却価格（万円）":
                gross_price_man,
            "売却諸費用（万円）":
                sale_cost_man,
            "概算売却手取り（万円）":
                net_price_man,
        }
    )


net_df = pd.DataFrame(
    net_rows
)


st.dataframe(
    net_df.style.format(
        {
            "予測売却価格（万円）":
                "{:,.0f}",
            "売却諸費用（万円）":
                "{:,.0f}",
            "概算売却手取り（万円）":
                "{:,.0f}",
        }
    ),
    use_container_width=True,
)


# =========================================================
# 注意書き
# =========================================================

st.warning(
    "将来価格は、築年数・金利・物価の入力シナリオに基づく参考値です。"
    "特に、学習データに存在しない高い物価指数や金利を入力した場合、"
    "予測の信頼性は低下します。"
)

st.caption(
    "本アプリの予測値は統計モデルによる参考値であり、"
    "鑑定評価、不動産会社の査定価格、将来の売却価格を"
    "保証するものではありません。"
)
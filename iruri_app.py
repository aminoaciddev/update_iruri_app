import re
import sqlite3
import hashlib
import uuid
import requests
import io
import os
from datetime import datetime, timedelta, timezone
import urllib.parse

import pandas as pd
import streamlit as st
import plotly.express as px


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="이루리 영어학원 성적분석", layout="wide")

# -------------------------
# CONFIG (중요)
# -------------------------
SPREADSHEET_ID = "18ffTcHQh2zO7kee7S-HYMnbNls8Qb0xrerkjJc0Dsfw"


APPS_SCRIPT_URL = st.secrets.get("APPS_SCRIPT_URL", "").strip()
APPS_SCRIPT_TOKEN = st.secrets.get("APPS_SCRIPT_TOKEN", "").strip()

st.write("APPS_SCRIPT_URL:", APPS_SCRIPT_URL[:40] + "...")
st.write("TOKEN len:", len(APPS_SCRIPT_TOKEN))


# -------------------------
# Paths & limits (Auth DB는 로컬 유지)
# -------------------------
DB_PATH = "data/auth.db"
PAIR_FAIL_LIMIT = 5
PAIR_LOCK_MIN = 10
IP_FAIL_LIMIT = 20
IP_LOCK_MIN = 30

# -------------------------
# Time / hash
# -------------------------
def now_utc():
    return datetime.now(timezone.utc)

def utc_ts():
    return datetime.now(timezone.utc).isoformat()

def hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -------------------------
# Text utilities
# -------------------------
def _norm_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()

def norm_key(x):
    s = _norm_str(x).replace("\u00A0", " ")
    s = re.sub(r"\s+", "", s)
    return s.strip()

def extract_first_number_str(x):
    s = _norm_str(x)
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else ""

def parse_percent_to_float(x):
    s = _norm_str(x)
    if s == "":
        return pd.NA
    m = re.search(r"(\d+(\.\d+)?)", s.replace(",", ""))
    if not m:
        return pd.NA
    try:
        return float(m.group(1))
    except Exception:
        return pd.NA

def to_int64_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def to_float(x):
    s = _norm_str(x)
    if s == "":
        return pd.NA
    m = re.search(r"(\d+(\.\d+)?)", s.replace(",", ""))
    if not m:
        return pd.NA
    try:
        return float(m.group(1))
    except Exception:
        return pd.NA

# -------------------------
# Question type mapping
# -------------------------
def build_question_type_map():
    m = {}
    def set_range(a, b, label):
        for q in range(a, b + 1):
            m[q] = label

    set_range(1, 17, "듣기")
    m[18] = "목적"
    m[19] = "심경"
    m[20] = "주장"
    m[21] = "함축적 의미"
    m[22] = "요지"
    m[23] = "주제"
    m[24] = "제목"
    m[25] = "표"
    m[26] = "지문 내용(세부)"
    set_range(27, 28, "실용자료(세부)")
    m[29] = "문법"
    m[30] = "어휘"
    set_range(31, 34, "빈칸 추론")
    m[35] = "글의 흐름"
    set_range(36, 37, "글의 순서")
    set_range(38, 39, "문장 삽입")
    m[40] = "문단 요약"
    m[41] = "제목(복합)"
    m[42] = "어휘(복합)"
    m[43] = "글의 순서(복합)"
    m[44] = "지칭 추론(복합)"
    m[45] = "내용일치/불일치(복합)"
    return m

QTYPE = build_question_type_map()

MAJOR_MAP = {
    "듣기": "듣기",
    "목적": "추론(단문)",
    "심경": "추론(단문)",
    "주장": "추론(단문)",
    "함축적 의미": "추론(단문)",
    "요지": "중심내용",
    "주제": "중심내용",
    "제목": "중심내용",
    "표": "세부정보",
    "지문 내용(세부)": "세부정보",
    "실용자료(세부)": "세부정보",
    "문법": "문법·어휘",
    "어휘": "문법·어휘",
    "어휘(복합)": "문법·어휘",
    "빈칸 추론": "빈칸 추론",
    "글의 흐름": "간접쓰기",
    "글의 순서": "간접쓰기",
    "문장 삽입": "간접쓰기",
    "문단 요약": "요약",
    "제목(복합)": "복합지문",
    "글의 순서(복합)": "복합지문",
    "지칭 추론(복합)": "복합지문",
    "내용일치/불일치(복합)": "복합지문",
}

MAJOR_COUNTS = {
    "듣기": 17,
    "추론(단문)": 4,
    "중심내용": 3,
    "세부정보": 4,
    "문법·어휘": 2,
    "빈칸 추론": 4,
    "간접쓰기": 5,
    "요약": 1,
    "복합지문": 5,
}

# -------------------------
# Wrong list parsing
# -------------------------
def parse_wrong_list(val):
    s = _norm_str(val).replace(" ", "")
    if s == "" or s == "미입력":
        return "미입력", [], []
    if s == "미응시":
        return "미응시", [], []
    if s in {"0", "없음"}:
        return "응시", [], []

    parts = s.split(",")
    wrong, invalid = [], []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"\d+", p):
            q = int(p)
            if 1 <= q <= 45:
                wrong.append(q)
            else:
                invalid.append(q)
    return "응시", sorted(set(wrong)), sorted(set(invalid))

def compute_major_counts(wrong_list):
    counts = {k: 0 for k in MAJOR_COUNTS.keys()}
    for q in wrong_list:
        detail = QTYPE.get(q, "기타")
        major = MAJOR_MAP.get(detail, "기타")
        if major in counts:
            counts[major] += 1
    return counts

# -------------------------
# Login attempt DB
# -------------------------
def ensure_auth_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # ✅ 폴더 없으면 생성
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            key TEXT PRIMARY KEY,
            fail_count INTEGER NOT NULL,
            first_fail_ts TEXT,
            last_fail_ts TEXT,
            locked_until_ts TEXT
        )
    """)
    conn.commit()
    return conn

def get_attempt(conn, key: str):
    cur = conn.cursor()
    cur.execute("SELECT fail_count, locked_until_ts FROM login_attempts WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return 0, None
    fail_count, locked_until_ts = row
    locked_until = None
    if locked_until_ts:
        try:
            locked_until = datetime.fromisoformat(locked_until_ts)
        except Exception:
            locked_until = None
    return fail_count, locked_until

def is_locked(conn, key: str):
    _, locked_until = get_attempt(conn, key)
    if locked_until and locked_until > now_utc():
        sec = int((locked_until - now_utc()).total_seconds())
        return True, sec
    return False, 0

def record_fail(conn, key: str, limit: int, lock_minutes: int):
    cur = conn.cursor()
    t = now_utc().isoformat()
    cur.execute("SELECT fail_count, locked_until_ts FROM login_attempts WHERE key=?", (key,))
    row = cur.fetchone()

    if not row:
        fail_count = 1
        locked_until = None
        if fail_count >= limit:
            locked_until = (now_utc() + timedelta(minutes=lock_minutes)).isoformat()
        cur.execute("""
            INSERT INTO login_attempts(key, fail_count, first_fail_ts, last_fail_ts, locked_until_ts)
            VALUES(?, ?, ?, ?, ?)
        """, (key, fail_count, t, t, locked_until))
    else:
        fail_count = int(row[0]) + 1
        locked_until_existing = row[1]
        locked_until_new = None
        if (not locked_until_existing) and fail_count >= limit:
            locked_until_new = (now_utc() + timedelta(minutes=lock_minutes)).isoformat()
        cur.execute("""
            UPDATE login_attempts
            SET fail_count=?, last_fail_ts=?, locked_until_ts=COALESCE(locked_until_ts, ?)
            WHERE key=?
        """, (fail_count, t, locked_until_new, key))

    conn.commit()

def reset_attempt(conn, key: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM login_attempts WHERE key=?", (key,))
    conn.commit()

def get_client_ip_best_effort():
    try:
        headers = st.context.headers
        xff = headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        xrip = headers.get("X-Real-Ip")
        if xrip:
            return xrip.strip()
    except Exception:
        pass
    return "unknown"

# -------------------------
# Read: gviz CSV
# -------------------------
def read_sheet_csv(sheet_name: str) -> pd.DataFrame:
    base = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq"
    params = {"tqx": "out:csv", "sheet": sheet_name}
    url = base + "?" + urllib.parse.urlencode(params)

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()

    txt = r.text.strip()
    if not txt:
        raise ValueError(f"{sheet_name}: CSV 내용이 비어있음")

    # ✅ 핵심: StringIO로 읽기
    df = pd.read_csv(
        io.StringIO(txt),
        encoding="utf-8",
    )
    return df
# -------------------------
# Write: Apps Script
# -------------------------
def _apps_script_post(payload: dict):
    if not APPS_SCRIPT_URL:
        raise RuntimeError("APPS_SCRIPT_URL이 설정되어 있지 않습니다. (secrets에 APPS_SCRIPT_URL 추가 필요)")
    if APPS_SCRIPT_TOKEN:
        payload = dict(payload)
        payload["token"] = APPS_SCRIPT_TOKEN

    r = requests.post(APPS_SCRIPT_URL, json=payload, timeout=15)
    r.raise_for_status()
    res = r.json()
    if not res.get("ok"):
        raise RuntimeError(res)
    return res

def append_wrong_answer_row(row_dict: dict):
    _apps_script_post({
        "action": "append",
        "sheet": "wrong_answer",
        "row": row_dict
    })
    st.cache_data.clear()

def update_wrong_answer_by_record_id(record_id: str, updates: dict):
    _apps_script_post({
        "action": "update",
        "sheet": "wrong_answer",
        "record_id": record_id,
        "updates": updates
    })
    st.cache_data.clear()

def delete_wrong_answer_by_record_id(record_id: str):
    _apps_script_post({
        "action": "delete",
        "sheet": "wrong_answer",
        "record_id": record_id
    })
    st.cache_data.clear()

def upsert_admin_solution(name: str, solution: str, updated_by: str = ""):
    _apps_script_post({
        "action": "upsert_admin_solution",
        "name": name,
        "solution": solution,
        "updated_at_utc": utc_ts(),
        "updated_by": updated_by
    })
    st.cache_data.clear()

# -------------------------
# Standardize ebsi/grammar (3학년만)
# -------------------------
def standardize_ebsi_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["출제기관", "회차", "학년", "문항번호", "전국오답률"])

    df = df.copy()
    colmap = {}
    for c in df.columns:
        cc = str(c).strip().replace("\u00A0", " ")
        cc_n = re.sub(r"\s+", "", cc)

        if cc_n in {"출제기관", "기관"}:
            colmap[c] = "출제기관"
        elif cc_n in {"회차", "시기", "시험", "모의고사"}:
            colmap[c] = "회차"
        elif cc_n in {"학년", "대상학년"}:
            colmap[c] = "학년"
        elif cc_n in {"문항번호", "문항", "번호"}:
            colmap[c] = "문항번호"
        elif cc_n in {"전국오답률", "전국오답률(%)", "오답률", "오답률(%)"}:
            colmap[c] = "전국오답률"

    df = df.rename(columns=colmap)

    for need in ["출제기관", "회차", "학년", "문항번호", "전국오답률"]:
        if need not in df.columns:
            df[need] = ""

    return df[["출제기관", "회차", "학년", "문항번호", "전국오답률"]].copy()

def standardize_grammar_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["출제기관", "회차", "학년", "문항번호", "정답개념"])

    df = df.copy()
    colmap = {}
    for c in df.columns:
        cc = str(c).strip().replace("\u00A0", " ")
        cc_n = re.sub(r"\s+", "", cc)

        if cc_n in {"출제기관", "기관"}:
            colmap[c] = "출제기관"
        elif cc_n in {"회차", "시기", "시험", "모의고사"}:
            colmap[c] = "회차"
        elif cc_n in {"학년", "대상학년"}:
            colmap[c] = "학년"
        elif cc_n in {"문항번호", "문항", "번호"}:
            colmap[c] = "문항번호"
        elif cc_n in {"정답개념", "개념", "정답개념명"}:
            colmap[c] = "정답개념"

    df = df.rename(columns=colmap)

    for need in ["출제기관", "회차", "학년", "문항번호", "정답개념"]:
        if need not in df.columns:
            df[need] = ""

    return df[["출제기관", "회차", "학년", "문항번호", "정답개념"]].copy()

# -------------------------
# Load all data (read only)
# -------------------------
@st.cache_data(show_spinner=False)
def load_data_from_gs():
    students = read_sheet_csv("students").dropna(how="all")
    wrong = read_sheet_csv("wrong_answer").dropna(how="all")
    admin_sol = read_sheet_csv("admin_solution").dropna(how="all")

    summaries = {
        "3": read_sheet_csv("3grade").dropna(how="all"),
        "2": read_sheet_csv("2grade").dropna(how="all"),
        "1": read_sheet_csv("1grade").dropna(how="all"),
    }

    # 3학년 전용
    try:
        ebsi_raw = read_sheet_csv("ebsi_stats").dropna(how="all")
    except Exception:
        ebsi_raw = pd.DataFrame()
    try:
        grammar_raw = read_sheet_csv("grammar_info").dropna(how="all")
    except Exception:
        grammar_raw = pd.DataFrame()

    # ---- validate base sheets ----
    required_students = {"student_id", "name", "grade", "role"}
    if not required_students.issubset(set(students.columns)):
        raise ValueError(f"students 컬럼 필요: {sorted(required_students)} / 현재: {list(students.columns)}")

    required_wrong = {"응시순서", "출제기관", "회차", "응시자", "원점수", "등급", "오답"}
    if not required_wrong.issubset(set(wrong.columns)):
        raise ValueError(f"wrong_answer 컬럼 필요: {sorted(required_wrong)} / 현재: {list(wrong.columns)}")

    # ---- students clean ----
    students = students.copy()
    students["name"] = students["name"].astype(str).str.strip()
    students["student_id"] = students["student_id"].astype(str).str.strip()
    students["role"] = students["role"].astype(str).str.strip()
    students["grade"] = students["grade"].astype(str).str.strip()

    # ---- wrong_answer clean ----
    wrong = wrong.copy()
    wrong["응시자"] = wrong["응시자"].astype(str).str.strip()
    wrong["출제기관"] = wrong["출제기관"].astype(str).str.strip()
    wrong["회차"] = wrong["회차"].astype(str).str.strip()

    # record columns optional
    for col in ["record_id", "created_at_utc", "updated_at_utc"]:
        if col not in wrong.columns:
            wrong[col] = ""

    statuses, wrong_lists, invalid_lists = [], [], []
    for _, r in wrong.iterrows():
        stt, wl, inv = parse_wrong_list(r["오답"])

        raw = _norm_str(r["원점수"])
        grd = _norm_str(r["등급"])

        if stt == "미입력":
            stt = "미입력" if (raw == "" and grd == "") else "응시"

        if _norm_str(r["원점수"]).replace(" ", "") == "미응시" or _norm_str(r["등급"]).replace(" ", "") == "미응시":
            stt, wl, inv = "미응시", [], []

        statuses.append(stt)
        wrong_lists.append(wl)
        invalid_lists.append(inv)

    wrong["status"] = statuses
    wrong["wrong_list"] = wrong_lists
    wrong["invalid_wrong_list"] = invalid_lists
    wrong["wrong_count"] = wrong["wrong_list"].apply(len)

    wrong["원점수_num"] = pd.to_numeric(wrong["원점수"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    wrong["등급_num"] = pd.to_numeric(wrong["등급"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    wrong["응시순서_num"] = pd.to_numeric(wrong["응시순서"], errors="coerce")

    df = wrong.merge(
        students[["student_id", "name", "grade", "role"]],
        left_on="응시자",
        right_on="name",
        how="left",
    )

    majors_df = pd.DataFrame(df["wrong_list"].apply(compute_major_counts).tolist())
    df = pd.concat([df.reset_index(drop=True), majors_df.reset_index(drop=True)], axis=1)

    # ---- summaries clean (이름_norm 추가) ----
    for k, s in summaries.items():
        if not s.empty and "이름" in s.columns:
            s = s.copy()
            s["이름_norm"] = s["이름"].astype(str).apply(lambda x: re.sub(r"\s+", "", x))
            summaries[k] = s

    # ---- 3학년 전용 normalize ----
    ebsi = standardize_ebsi_columns(ebsi_raw)
    ebsi = ebsi.copy()
    ebsi["출제기관_key"] = ebsi["출제기관"].apply(norm_key)
    ebsi["회차_key"] = ebsi["회차"].apply(norm_key)
    ebsi["학년_key"] = ebsi["학년"].apply(extract_first_number_str)
    ebsi["문항번호_num"] = to_int64_series(ebsi["문항번호"])
    ebsi["전국오답률_num"] = ebsi["전국오답률"].apply(parse_percent_to_float)

    grammar = standardize_grammar_columns(grammar_raw)
    grammar = grammar.copy()
    grammar["출제기관_key"] = grammar["출제기관"].apply(norm_key)
    grammar["회차_key"] = grammar["회차"].apply(norm_key)
    grammar["학년_key"] = grammar["학년"].apply(extract_first_number_str)
    grammar["문항번호_num"] = to_int64_series(grammar["문항번호"])
    grammar["정답개념"] = grammar["정답개념"].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
    grammar["정답개념_카테고리"] = grammar["정답개념"].astype(str).str.split("(", n=1).str[0].str.strip()

    # ---- admin_solution clean ----
    admin_sol = admin_sol.copy()
    if "name" not in admin_sol.columns: admin_sol["name"] = ""
    if "solution" not in admin_sol.columns: admin_sol["solution"] = ""
    admin_sol["name"] = admin_sol["name"].astype(str).str.strip()
    admin_sol["solution"] = admin_sol["solution"].astype(str).fillna("").str.strip()

    return students, df, ebsi, grammar, summaries, admin_sol

# -------------------------
# Login screen
# -------------------------
def render_login(students: pd.DataFrame):
    st.markdown("<h1 style='text-align:center;'>이루리 영어학원 방학 모의고사 성적분석</h1>", unsafe_allow_html=True)
    st.write("")

    name = st.text_input("이름")
    sid = st.text_input("고유번호", type="password")

    conn = ensure_auth_db()

    st.write("")
    if st.button("들어가기", type="primary"):
        if not name or not sid:
            st.error("이름과 고유번호를 모두 입력하세요.")
            return

        name_norm = str(name).strip()
        sid_norm = str(sid).strip()

        pair_key = hash_key(f"pair::{name_norm}::{sid_norm}")
        ip = get_client_ip_best_effort()
        ip_key = hash_key(f"ip::{ip}")

        locked, sec = is_locked(conn, pair_key)
        if locked:
            st.error(f"로그인 시도가 너무 많아서 잠겼습니다. {sec//60}분 {sec%60}초 후에 다시 시도하세요.")
            return

        locked, sec = is_locked(conn, ip_key)
        if locked and ip != "unknown":
            st.error(f"접속 시도가 너무 많아서 잠겼습니다. {sec//60}분 {sec%60}초 후에 다시 시도하세요.")
            return

        matched = students[
            (students["name"].astype(str).str.strip() == name_norm) &
            (students["student_id"].astype(str).str.strip() == sid_norm)
        ]

        if matched.empty:
            record_fail(conn, pair_key, PAIR_FAIL_LIMIT, PAIR_LOCK_MIN)
            if ip != "unknown":
                record_fail(conn, ip_key, IP_FAIL_LIMIT, IP_LOCK_MIN)
            st.error("이름 또는 고유번호를 확인해주세요.")
            return

        reset_attempt(conn, pair_key)
        if ip != "unknown":
            reset_attempt(conn, ip_key)

        role = str(matched.iloc[0]["role"]).strip()
        grade = matched.iloc[0].get("grade", None)

        st.session_state["logged_in"] = True
        st.session_state["role"] = role
        st.session_state["name"] = name_norm
        st.session_state["student_id"] = sid_norm
        st.session_state["grade"] = grade

        st.session_state["student_panel"] = "none"  # none | total | exam
        st.session_state["admin_mode"] = "관리자 대시보드"

        st.rerun()

# -------------------------
# Student dashboard
# -------------------------
def render_student_dashboard(df, ebsi, grammar, summaries, admin_sol, name, grade, is_preview=False):
    grade_num = extract_first_number_str(grade)
    search_name = re.sub(r"\s+", "", str(name))

    suffix = " (미리보기)" if is_preview else ""

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"### {name}{suffix}")
    with h2:
        st.markdown(
            f"<div style='text-align:right; font-size:16px; margin-top:8px;'>학년: <b>{grade}</b></div>",
            unsafe_allow_html=True
        )
    
    me = df[df["응시자"] == name].copy()
    me = me.sort_values(["응시순서_num", "응시순서"], na_position="last")
    taken = me[me["status"] == "응시"].copy()

    # 등급 추이
    st.markdown("#### 회차별 등급 추이")
    chart_df = taken.dropna(subset=["등급_num", "응시순서_num"]).copy()
    if chart_df.empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig = px.line(chart_df, x="응시순서_num", y="등급_num", markers=True,
                      hover_data=["출제기관", "회차", "원점수_num", "wrong_count"])
        fig.update_layout(showlegend=False)
        fig.update_yaxes(autorange="reversed", dtick=1, range=[9.5, 0.5], title="등급")
        fig.update_xaxes(dtick=1, title="회차(응시순서)")
        st.plotly_chart(fig, use_container_width=True)

    # KPI (학년별 요약시트)
    summary_df = summaries.get(grade_num, pd.DataFrame())
    total_cnt, avg_grade, l_val, r_val = "-", "-", "-", "-"

    if not summary_df.empty and "이름_norm" in summary_df.columns:
        match = summary_df[summary_df["이름_norm"] == search_name]
        if not match.empty:
            r = match.iloc[0]
            total_cnt = _norm_str(r.get("모의고사응시횟수", "-"))
            avg_grade = _norm_str(r.get("등급평균", "-"))

            l_val = _norm_str(r.get("듣기영역(1~17번)", "-"))
            r_val = _norm_str(r.get("독해영역(18~45번)", "-"))


    st.markdown("""
    <style>
    .kpi-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #f0f2f6;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label { font-size: 20px; color: #555; margin-bottom: 10px; font-weight: 600; }
    .kpi-value { font-size: 22px; font-weight: 800; color: #1f77b4; word-break: break-all; }

    .flow-arrow {
        text-align: center;
        font-size: 35px;
        color: #1f77b4;
        margin: 20px 0;
        font-weight: bold;
        line-height: 1;
    }

    .solution-box {
        border: 2px solid #1f77b4;
        border-radius: 15px;
        padding: 20px;
        background-color: #f0f8ff;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f'<div class="kpi-container"><div class="kpi-label">모의고사 응시 횟수</div><div class="kpi-value">{total_cnt}회</div></div>',
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f'<div class="kpi-container"><div class="kpi-label">듣기영역(1~17번)</div><div class="kpi-value">{l_val}</div></div>',
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f'<div class="kpi-container"><div class="kpi-label">독해영역(18~45번)</div><div class="kpi-value">{r_val}</div></div>',
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f'<div class="kpi-container"><div class="kpi-label">등급 평균</div><div class="kpi-value">{avg_grade}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # 솔루션
    sol_row = admin_sol[admin_sol["name"].astype(str).str.strip() == str(name).strip()].head(1)
    sol_text = _norm_str(sol_row.iloc[0].get("solution", "")) if not sol_row.empty else ""
    st.markdown("#### 🟦 솔루션")
    if sol_text.strip() == "":
        st.markdown('<div class="solution-box" style="color:#999;">작성된 솔루션이 없습니다.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="solution-box">{sol_text}</div>', unsafe_allow_html=True)




    st.divider()

    # 버튼 패널
    b1, b2 = st.columns(2)
    with b1:
        if st.button("총 오답 현황 확인하기", key=f"btn_total_{name}"):
            st.session_state["student_panel"] = "total" if st.session_state.get("student_panel") != "total" else "none"
    with b2:
        if st.button("특정 회차 오답 확인하기", key=f"btn_exam_{name}"):
            st.session_state["student_panel"] = "exam" if st.session_state.get("student_panel") != "exam" else "none"

    panel = st.session_state.get("student_panel", "none")

    if panel == "total":
        st.markdown("### 총 오답 현황 (회차별)")
        show = me[["응시순서", "출제기관", "회차", "status", "원점수_num", "등급_num", "wrong_count", "오답"]].copy()
        st.dataframe(show, use_container_width=True)

    if panel == "exam":
        st.markdown("### 특정 회차 오답")
        exams = taken[["응시순서_num", "응시순서", "출제기관", "회차", "wrong_list"]].copy()
        exams = exams.sort_values(["응시순서_num", "응시순서"])
        if exams.empty:
            st.caption("응시 기록이 없습니다.")
        else:
            labels = [f"{int(r['응시순서'])}. {r['출제기관']} / {r['회차']}" for _, r in exams.iterrows()]
            choice = st.selectbox("회차", labels, key=f"exam_select_{name}")
            row = exams.iloc[labels.index(choice)]
            wl = row["wrong_list"]

            if not wl:
                st.success("틀린 문제가 없습니다!")
            else:
                # 3학년만 상세표
                if grade_num == "3":
                    org = row["출제기관"]
                    rnd = row["회차"]
                    tbl = pd.DataFrame({"문항번호": wl})
                    tbl["유형"] = tbl["문항번호"].map(lambda q: QTYPE.get(int(q), "기타"))
                    tbl["대분류"] = tbl["유형"].map(lambda t: MAJOR_MAP.get(t, "기타"))

                    org_k = norm_key(org)
                    rnd_k = norm_key(rnd)
                    g_k = "3"

                    base = ebsi[(ebsi["출제기관_key"] == org_k) & (ebsi["회차_key"] == rnd_k)].copy()
                    use = base
                    base_g = base[base["학년_key"] == g_k]
                    if not base_g.empty:
                        use = base_g

                    if not use.empty:
                        rate_sub = use[["문항번호_num", "전국오답률_num"]].rename(columns={"문항번호_num": "문항번호"})
                        tbl = tbl.merge(rate_sub, on="문항번호", how="left")
                        tbl["전국오답률(%)"] = tbl["전국오답률_num"].apply(lambda x: "-" if pd.isna(x) else round(float(x), 1))
                        tbl.drop(columns=["전국오답률_num"], inplace=True, errors="ignore")
                    else:
                        tbl["전국오답률(%)"] = "-"

                    gbase = grammar[(grammar["출제기관_key"] == org_k) & (grammar["회차_key"] == rnd_k)].copy()
                    guse = gbase
                    gbase_g = gbase[gbase["학년_key"] == g_k]
                    if not gbase_g.empty:
                        guse = gbase_g

                    if not guse.empty:
                        gsub = guse[["문항번호_num", "정답개념_카테고리", "정답개념"]].rename(columns={"문항번호_num": "문항번호"})
                        tbl = tbl.merge(gsub, on="문항번호", how="left")
                    else:
                        tbl["정답개념_카테고리"] = "-"
                        tbl["정답개념"] = "-"

                    desired = ["문항번호", "유형", "대분류", "전국오답률(%)", "정답개념_카테고리", "정답개념"]
                    for c in desired:
                        if c not in tbl.columns:
                            tbl[c] = "-"
                    st.dataframe(tbl[desired], use_container_width=True)
                else:
                    st.write(f"**틀린 문항 번호:** {', '.join(map(str, wl))}")
                    st.caption("※ 1, 2학년은 상세 오답 통계를 제공하지 않습니다.")

    # 누적 취약 유형
    st.divider()
    st.markdown("### 취약 유형 (누적 대분류)")

    if taken.empty:
        st.info("응시 데이터가 없어서 분석할 수 없습니다.")
        return

    major_cols = list(MAJOR_COUNTS.keys())
    sums = taken[major_cols].sum().sort_values(ascending=False)
    st.bar_chart(sums)

    # 3학년만 문법 키워드
    if grade_num == "3":
        gram_vocab_wrong = int(sums.get("문법·어휘", 0))
        if gram_vocab_wrong > 10:
            st.markdown("#### ⚠️ 문법·어휘 오답이 많습니다 (10개 초과)")

            all_wrong = []
            for wl in taken["wrong_list"]:
                all_wrong.extend(wl)

            gv_set = set([29, 30, 42])
            gv_wrong = [q for q in all_wrong if q in gv_set]

            if not gv_wrong:
                st.caption("문법·어휘로 분류된 오답은 있으나, 문항번호(29/30/42)에서 직접 확인되지 않았습니다.")
            else:
                cats = []
                for _, row in taken.iterrows():
                    org_k = norm_key(row["출제기관"])
                    rnd_k = norm_key(row["회차"])

                    gbase = grammar[(grammar["출제기관_key"] == org_k) & (grammar["회차_key"] == rnd_k)].copy()
                    guse = gbase[gbase["학년_key"] == "3"]
                    if guse.empty:
                        guse = gbase

                    if guse.empty:
                        continue

                    wrongs = set(row["wrong_list"])
                    sub = guse[guse["문항번호_num"].isin(list(wrongs))][["정답개념_카테고리"]].copy()
                    for v in sub["정답개념_카테고리"].dropna().astype(str).tolist():
                        vv = v.strip()
                        if vv and vv != "-":
                            cats.append(vv)

                if not cats:
                    st.caption("grammar_info에 매핑된 '정답개념' 데이터가 부족해서 키워드를 만들 수 없습니다.")
                else:
                    top = pd.Series(cats).value_counts().head(8)
                    st.write( "**틀린 문법 개념 키워드(상위):** "  + " · ".join([f"{idx}({int(val)})" for idx, val in top.items()]))


# -------------------------
# Admin dashboard
# -------------------------
def render_admin_dashboard(df: pd.DataFrame, students_df: pd.DataFrame, admin_sol: pd.DataFrame):
    st.markdown("### 👨‍🏫 관리자 모드")
    st.caption("학생 기록 관리 및 학원 전체 통계 (구글시트 + Apps Script 쓰기)")

    taken = df[df["status"] == "응시"].copy()
    absent = df[df["status"] == "미응시"].copy()
    missing = df[df["status"] == "미입력"].copy()

    k1, k2, k3 = st.columns(3)
    k1.metric("총 응시 기록", len(taken))
    k2.metric("미응시(결석)", len(absent))
    k3.metric("데이터 미입력", len(missing))

    st.divider()

    # 솔루션 작성
    st.markdown("#### 📝 학생별 관리자 솔루션 작성")
    students_list = sorted(students_df["name"].dropna().astype(str).str.strip().unique().tolist())
    target_student = st.selectbox("솔루션을 작성할 학생 선택", students_list, key="admin_sol_select")

    current_sol_row = admin_sol[admin_sol["name"].astype(str).str.strip() == str(target_student).strip()].head(1)
    current_text = _norm_str(current_sol_row.iloc[0].get("solution", "")) if not current_sol_row.empty else ""

    new_sol_text = st.text_area("솔루션 내용", value=current_text, height=150,
                               placeholder="예: 빈칸 추론 유형의 오답률이 높습니다. 단어장 5과를 복습하세요.")

    if st.button("솔루션 저장", type="primary"):
        try:
            upsert_admin_solution(target_student, new_sol_text, updated_by=str(st.session_state.get("name","")))
            st.success("저장 완료!")
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    st.divider()

    # 응시 기록 관리
    st.markdown("#### 📋 응시 기록 관리")
    tab1, tab2 = st.tabs(["🆕 기록 추가", "⚙️ 기록 수정/삭제"])

    with tab1:
        with st.form("add_exam_form_gs", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            new_order = c1.text_input("응시순서(숫자)", placeholder="예: 5")
            org_choice = c2.selectbox("출제기관", ["평가원", "교육청", "사설", "기타"])
            new_org = c2.text_input("직접 입력(기타 선택 시)") if org_choice == "기타" else org_choice
            new_round = c3.text_input("회차", placeholder="예: 25년 3월")

            new_name = st.selectbox("응시자 이름", students_list)

            c4, c5 = st.columns(2)
            new_score = c4.text_input("원점수", placeholder="예: 92")
            new_grade = c5.text_input("등급", placeholder="예: 2")
            new_wrong = st.text_input("오답(쉼표 구분)", placeholder="예: 29, 31, 34 / 없으면 0 / 미응시는 '미응시'")

            if st.form_submit_button("기록 추가 저장"):
                if not new_order.strip().isdigit():
                    st.error("응시순서는 숫자만 입력 가능합니다.")
                else:
                    rid = str(uuid.uuid4())
                    now = utc_ts()
                    row_dict = {
                        "record_id": rid,
                        "created_at_utc": now,
                        "updated_at_utc": now,
                        "응시순서": new_order.strip(),
                        "출제기관": new_org.strip(),
                        "회차": new_round.strip(),
                        "응시자": new_name.strip(),
                        "원점수": new_score.strip(),
                        "등급": new_grade.strip(),
                        "오답": new_wrong.strip(),
                    }
                    try:
                        append_wrong_answer_row(row_dict)
                        st.success("기록이 추가되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

    with tab2:
        edit_name = st.selectbox("조회할 학생", students_list, key="edit_name_select")
        student_records = taken[taken["응시자"] == edit_name].copy()

        if student_records.empty:
            st.info("해당 학생의 응시 기록이 없습니다.")
        else:
            # record_id가 있어야 수정/삭제가 안정적
            if "record_id" not in student_records.columns:
                student_records["record_id"] = ""

            record_labels = []
            label_to_rid = {}
            for _, r in student_records.iterrows():
                rid = str(r.get("record_id", "")).strip()
                label = f"{r['출제기관']} | {r['회차']} ({r['원점수']}점)"
                if rid:
                    label += f"  [{rid[:8]}]"
                record_labels.append(label)
                label_to_rid[label] = rid

            selected_label = st.selectbox("수정/삭제할 기록", record_labels)
            selected_rid = label_to_rid.get(selected_label, "")

            record_data = student_records[student_records["record_id"].astype(str).str.strip() == selected_rid].head(1)
            if record_data.empty:
                record_data = student_records.iloc[[record_labels.index(selected_label)]]
            record_data = record_data.iloc[0]

            with st.form("edit_exam_form_gs"):
                c1, c2 = st.columns(2)
                up_score = c1.text_input("원점수 수정", value=str(record_data.get("원점수","")))
                up_grade = c2.text_input("등급 수정", value=str(record_data.get("등급","")))
                up_wrong = st.text_input("오답 수정", value=str(record_data.get("오답","")))
                up_order = st.text_input("응시순서 수정", value=str(record_data.get("응시순서","")))

                btn_up, btn_del = st.columns(2)
                if btn_up.form_submit_button("수정 완료", type="primary"):
                    try:
                        rid = str(record_data.get("record_id","")).strip()
                        if not rid:
                            st.error("record_id가 없어 수정이 불가능합니다. (기록을 새로 추가하거나 시트에 record_id 컬럼을 추가해 주세요)")
                        else:
                            updates = {
                                "원점수": up_score.strip(),
                                "등급": up_grade.strip(),
                                "오답": up_wrong.strip(),
                                "응시순서": up_order.strip(),
                                "updated_at_utc": utc_ts(),
                            }
                            update_wrong_answer_by_record_id(rid, updates)
                            st.success("수정되었습니다.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"수정 실패: {e}")

                if btn_del.form_submit_button("기록 삭제", type="secondary"):
                    try:
                        rid = str(record_data.get("record_id","")).strip()
                        if not rid:
                            st.error("record_id가 없어 삭제가 불가능합니다. (시트에 record_id 컬럼 필요)")
                        else:
                            delete_wrong_answer_by_record_id(rid)
                            st.warning("기록이 삭제되었습니다.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"삭제 실패: {e}")

    st.divider()

    # 학생별 조회
    st.markdown("#### 🔍 학생별 데이터 전체 조회")
    view_student = st.selectbox("학생 선택", students_list, key="admin_view_select")
    if view_student:
        sub_df = df[df["응시자"] == view_student].sort_values("응시순서_num")
        st.dataframe(sub_df[["응시순서", "출제기관", "회차", "status", "원점수", "등급", "오답", "wrong_count"]],
                     use_container_width=True)

# -------------------------
# Main
# -------------------------
def main():
    try:
        students, df, ebsi, grammar, summaries, admin_sol = load_data_from_gs()
    except Exception as e:
        st.error(f"구글 시트 연동 실패: {e}")
        st.stop()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        render_login(students)
        return

    role = str(st.session_state.get("role", "")).strip()
    name = st.session_state.get("name")
    grade = st.session_state.get("grade")

    with st.sidebar:
        st.markdown("## 계정")
        st.write(f"- 이름: **{st.session_state.get('name')}**")
        st.write(f"- 역할: **{st.session_state.get('role')}**")

        if role == "admin":
            st.markdown("## 관리자 메뉴")
            st.session_state["admin_mode"] = st.radio(
                "화면 선택",
                ["관리자 대시보드", "학생 화면 미리보기"],
                index=0 if st.session_state.get("admin_mode") != "학생 화면 미리보기" else 1
            )

        if st.button("로그아웃"):
            st.session_state.clear()
            st.rerun()

    if role == "admin":
        mode = st.session_state.get("admin_mode", "관리자 대시보드")
        if mode == "관리자 대시보드":
            render_admin_dashboard(df, students, admin_sol)
        else:
            st.markdown("### 학생 화면 미리보기(관리자)")
            students_list = sorted(students["name"].dropna().astype(str).str.strip().unique().tolist())
            preview_name = st.selectbox("미리볼 학생 선택", students_list, key="preview_student")

            g = students[students["name"].astype(str).str.strip() == str(preview_name).strip()]
            preview_grade = g.iloc[0]["grade"] if not g.empty else ""

            render_student_dashboard(df, ebsi, grammar, summaries, admin_sol,
                                     name=preview_name, grade=preview_grade, is_preview=True)
    else:
        render_student_dashboard(df, ebsi, grammar, summaries, admin_sol,
                                 name=name, grade=grade, is_preview=False)

if __name__ == "__main__":
    main()


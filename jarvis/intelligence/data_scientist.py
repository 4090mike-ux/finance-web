"""
JARVIS 자율 데이터 과학자 — Iteration 11
데이터 파일을 받아 자동으로 분석하고 인사이트를 생성한다

영감:
  - AutoML (자동 머신러닝)
  - Pandas Profiling / ydata-profiling
  - 탐색적 데이터 분석 (EDA)
  - 자연어 데이터 쿼리 (NL2SQL)

핵심 개념:
  JARVIS는 CSV 한 줄로 전체 데이터셋을 이해한다
  자동 통계 요약 → 이상치 탐지 → 패턴 인식 → 시각화 제안
  LLM이 데이터를 보고 자연어로 인사이트를 설명한다
"""

import json
import time
import logging
import threading
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")   # GUI 없이 파일로 저장
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class DataProfile:
    """데이터셋 프로파일"""
    file_name: str
    row_count: int
    col_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_stats: Dict[str, Dict]      # 수치형 컬럼 통계
    categorical_stats: Dict[str, Dict]  # 범주형 컬럼 통계
    correlations: List[Tuple[str, str, float]]  # 강한 상관관계
    anomalies: List[str]                # 이상치 발견
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "file_name": self.file_name,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "missing_values": self.missing_values,
            "numeric_stats": self.numeric_stats,
            "categorical_stats": self.categorical_stats,
            "correlations": [
                {"col1": c[0], "col2": c[1], "correlation": c[2]}
                for c in self.correlations
            ],
            "anomalies": self.anomalies,
        }


@dataclass
class AnalysisResult:
    """분석 결과"""
    dataset_name: str
    question: str
    answer: str                         # 자연어 답변
    data_points: List[Dict]             # 핵심 데이터 포인트
    charts_generated: List[str]         # 생성된 차트 파일 경로
    recommendations: List[str]          # 권장 사항
    confidence: float = 0.85
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "question": self.question,
            "answer": self.answer,
            "data_points": self.data_points,
            "charts_generated": self.charts_generated,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
        }


# ════════════════════════════════════════════════════════════════
# 자율 데이터 과학자
# ════════════════════════════════════════════════════════════════

class DataScientist:
    """
    JARVIS 자율 데이터 과학자
    CSV / JSON / 텍스트 데이터를 받아 자동으로 분석하고
    자연어로 인사이트를 제공한다
    """

    def __init__(self, llm_manager, charts_dir: str = "data/charts"):
        self.llm = llm_manager
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_datasets: Dict[str, Any] = {}   # name → DataFrame
        self._profiles: Dict[str, DataProfile] = {}
        self._lock = threading.Lock()
        self._stats = {
            "datasets_loaded": 0,
            "analyses_run": 0,
            "charts_created": 0,
            "queries_answered": 0,
        }
        logger.info(f"DataScientist initialized — pandas={PANDAS_AVAILABLE}, matplotlib={MATPLOTLIB_AVAILABLE}")

    # ── 데이터 로딩 ──────────────────────────────────────────────

    def load_file(self, file_path: str, name: Optional[str] = None) -> Dict:
        """파일에서 데이터셋 로드 (CSV, JSON, TSV, Excel 지원)"""
        if not PANDAS_AVAILABLE:
            return {"success": False, "error": "pandas 패키지가 없습니다. pip install pandas"}

        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"파일 없음: {file_path}"}

        dataset_name = name or path.stem
        ext = path.suffix.lower()

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8-sig")
            elif ext in (".json", ".jsonl"):
                df = pd.read_json(file_path)
            elif ext == ".tsv":
                df = pd.read_csv(file_path, sep="\t")
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(file_path)
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                return {"success": False, "error": f"지원하지 않는 형식: {ext}"}

            with self._lock:
                self._loaded_datasets[dataset_name] = df
                self._stats["datasets_loaded"] += 1

            profile = self._profile_dataframe(df, path.name)
            self._profiles[dataset_name] = profile

            logger.info(f"Dataset loaded: '{dataset_name}' ({df.shape[0]}행 × {df.shape[1]}열)")
            return {
                "success": True,
                "name": dataset_name,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "profile": profile.to_dict(),
            }

        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return {"success": False, "error": str(e)}

    def load_from_dict(self, data: List[Dict], name: str) -> Dict:
        """딕셔너리 리스트에서 데이터셋 생성"""
        if not PANDAS_AVAILABLE:
            return {"success": False, "error": "pandas 없음"}
        try:
            df = pd.DataFrame(data)
            with self._lock:
                self._loaded_datasets[name] = df
                self._stats["datasets_loaded"] += 1
            profile = self._profile_dataframe(df, name)
            self._profiles[name] = profile
            return {"success": True, "name": name, "shape": list(df.shape)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── 자동 프로파일링 ──────────────────────────────────────────

    def _profile_dataframe(self, df, file_name: str) -> DataProfile:
        """데이터프레임 자동 프로파일 생성"""
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}

        # 수치형 통계
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            numeric_stats[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
            }

        # 범주형 통계
        categorical_stats = {}
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols[:10]:  # 최대 10개
            vc = df[col].value_counts()
            categorical_stats[col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(5).items()},
                "null_count": int(df[col].isna().sum()),
            }

        # 상관관계 (수치형 컬럼 간)
        correlations = []
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) > 0.7:  # 강한 상관관계만
                            correlations.append((col1, col2, round(float(corr_val), 3)))
            except Exception:
                pass

        # 이상치 탐지 (IQR 방법)
        anomalies = []
        for col in numeric_cols[:5]:
            try:
                series = df[col].dropna()
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = int(((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum())
                if outlier_count > 0:
                    pct = round(outlier_count / len(series) * 100, 1)
                    anomalies.append(f"'{col}': {outlier_count}개 이상치 ({pct}%)")
            except Exception:
                pass

        return DataProfile(
            file_name=file_name,
            row_count=len(df),
            col_count=len(df.columns),
            columns=list(df.columns),
            dtypes=dtypes,
            missing_values=missing,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            correlations=correlations,
            anomalies=anomalies,
        )

    # ── 자연어 쿼리 ──────────────────────────────────────────────

    def ask(self, dataset_name: str, question: str) -> AnalysisResult:
        """
        자연어 질문으로 데이터 분석
        예: "매출이 가장 높은 달은?" "이상치가 있나요?" "상관관계를 보여줘"
        """
        start = time.time()

        df = self._loaded_datasets.get(dataset_name)
        if df is None:
            return self._error_result(dataset_name, question, f"데이터셋 '{dataset_name}'이 로드되지 않았습니다")

        profile = self._profiles.get(dataset_name)
        profile_summary = self._profile_to_summary(profile) if profile else "프로파일 없음"

        # 데이터 샘플 (첫 10행)
        try:
            sample_data = df.head(10).to_string(index=False)
        except Exception:
            sample_data = "(샘플 없음)"

        # 코드 생성 프롬프트
        code_prompt = f"""다음 데이터셋에 대한 질문에 답하기 위한 Python/pandas 코드를 작성하세요.

데이터셋 정보:
{profile_summary}

데이터 샘플 (첫 10행):
{sample_data}

질문: {question}

규칙:
1. pandas 변수명은 반드시 `df`를 사용하세요
2. 결과를 `result` 변수에 저장하세요
3. 출력이 필요하면 print() 대신 result 변수를 사용하세요
4. matplotlib 차트가 필요하면 plt.savefig('chart.png', dpi=100, bbox_inches='tight')를 사용하세요
5. 안전한 코드만 작성하세요 (파일 쓰기/시스템 접근 금지)

Python 코드만 출력하세요 (설명 없음, 마크다운 코드블록 없음):"""

        try:
            code = self.llm.generate(code_prompt, max_tokens=800)
            if isinstance(code, dict):
                code = code.get("content", "")
            # 코드블록 정리
            code = re.sub(r"```python\s*", "", code)
            code = re.sub(r"```\s*", "", code)
            code = code.strip()
        except Exception as e:
            code = f"result = 'LLM 코드 생성 실패: {e}'"

        # 코드 실행 (샌드박스)
        exec_result = self._safe_execute(code, df)
        charts = exec_result.get("charts", [])
        exec_output = exec_result.get("result", "실행 결과 없음")

        # 자연어 해석
        interpret_prompt = f"""데이터 분석 결과를 한국어로 명확하게 설명하세요.

질문: {question}

분석 결과:
{str(exec_output)[:2000]}

규칙:
- 핵심 수치와 인사이트를 구체적으로 언급
- 2-4문장으로 간결하게
- 비전문가도 이해할 수 있는 언어 사용
- 추가 권장 분석이 있다면 1가지만 제안"""

        try:
            answer = self.llm.generate(interpret_prompt, max_tokens=300)
            if isinstance(answer, dict):
                answer = answer.get("content", str(answer))
        except Exception as e:
            answer = f"결과: {str(exec_output)[:500]}"

        # 데이터 포인트 추출
        data_points = self._extract_data_points(exec_output)
        recommendations = self._generate_recommendations(question, exec_output, profile)

        with self._lock:
            self._stats["queries_answered"] += 1
            self._stats["analyses_run"] += 1
            self._stats["charts_created"] += len(charts)

        result = AnalysisResult(
            dataset_name=dataset_name,
            question=question,
            answer=answer,
            data_points=data_points,
            charts_generated=charts,
            recommendations=recommendations,
            processing_time=time.time() - start,
        )
        logger.info(f"Query answered: '{question[:40]}...' [{result.processing_time:.1f}s]")
        return result

    def auto_eda(self, dataset_name: str) -> Dict:
        """전체 EDA 자동 실행"""
        df = self._loaded_datasets.get(dataset_name)
        if df is None:
            return {"error": f"데이터셋 '{dataset_name}'이 없습니다"}

        profile = self._profiles.get(dataset_name)
        results = []

        # 자동 분석 질문 목록
        auto_questions = [
            "이 데이터셋의 전반적인 특성과 품질을 평가해주세요",
            "수치형 컬럼들의 분포와 이상치를 분석해주세요",
            "가장 강한 상관관계를 가진 변수들을 찾아주세요",
        ]

        if profile and profile.categorical_stats:
            auto_questions.append("범주형 변수들의 분포를 분석해주세요")

        for q in auto_questions:
            result = self.ask(dataset_name, q)
            results.append(result.to_dict())

        # 종합 인사이트
        summary_prompt = f"""다음은 데이터셋 '{dataset_name}'의 자동 분석 결과입니다:

{chr(10).join([r['answer'] for r in results])}

이를 종합하여 3가지 핵심 인사이트와 2가지 개선 권장사항을 제공하세요."""

        try:
            summary = self.llm.generate(summary_prompt, max_tokens=400)
            if isinstance(summary, dict):
                summary = summary.get("content", "")
        except Exception:
            summary = "요약 생성 실패"

        return {
            "dataset": dataset_name,
            "profile": profile.to_dict() if profile else {},
            "analyses": results,
            "summary": summary,
            "charts": [c for r in results for c in r.get("charts_generated", [])],
        }

    # ── 차트 생성 ────────────────────────────────────────────────

    def create_chart(self, dataset_name: str, chart_type: str,
                     x_col: str, y_col: Optional[str] = None,
                     title: str = "") -> Optional[str]:
        """차트 직접 생성"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        df = self._loaded_datasets.get(dataset_name)
        if df is None:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            chart_type_lower = chart_type.lower()

            if chart_type_lower == "bar" and y_col:
                df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)
            elif chart_type_lower == "line" and y_col:
                df.plot(x=x_col, y=y_col, kind="line", ax=ax)
            elif chart_type_lower == "scatter" and y_col:
                ax.scatter(df[x_col], df[y_col], alpha=0.5)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            elif chart_type_lower == "histogram":
                df[x_col].hist(ax=ax, bins=30)
                ax.set_xlabel(x_col)
                ax.set_ylabel("빈도")
            elif chart_type_lower == "pie":
                df[x_col].value_counts().head(8).plot(kind="pie", ax=ax, autopct="%1.1f%%")
            elif chart_type_lower == "box" and y_col:
                df.boxplot(column=y_col, by=x_col, ax=ax)
            else:
                df[x_col].hist(ax=ax, bins=20)

            ax.set_title(title or f"{chart_type}: {x_col}" + (f" vs {y_col}" if y_col else ""))
            plt.tight_layout()

            import uuid
            chart_path = str(self.charts_dir / f"chart_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            with self._lock:
                self._stats["charts_created"] += 1
            return chart_path

        except Exception as e:
            logger.warning(f"차트 생성 실패: {e}")
            plt.close("all")
            return None

    # ── 안전 실행 ────────────────────────────────────────────────

    def _safe_execute(self, code: str, df) -> Dict:
        """안전 샌드박스에서 pandas 코드 실행"""
        charts_created = []
        import uuid

        # 차트 저장 경로 주입
        chart_path = str(self.charts_dir / f"chart_{uuid.uuid4().hex[:8]}.png")

        # 금지 패턴
        forbidden = ["os.system", "subprocess", "exec(", "eval(", "open(",
                     "__import__", "shutil", "rmdir", "remove", "unlink"]
        for pattern in forbidden:
            if pattern in code:
                return {"result": f"금지된 코드 패턴: {pattern}", "charts": []}

        local_vars = {"df": df, "result": None, "chart_path": chart_path}
        if PANDAS_AVAILABLE:
            import pandas as pd
            import numpy as np
            local_vars["pd"] = pd
            local_vars["np"] = np

        if MATPLOTLIB_AVAILABLE:
            local_vars["plt"] = plt

        # chart.png → 실제 경로로 치환
        code = code.replace("'chart.png'", f"'{chart_path}'")
        code = code.replace('"chart.png"', f'"{chart_path}"')

        try:
            exec(code, {"__builtins__": {"print": print, "len": len, "range": range,
                                          "list": list, "dict": dict, "str": str,
                                          "int": int, "float": float, "round": round,
                                          "sorted": sorted, "sum": sum, "min": min,
                                          "max": max, "abs": abs, "enumerate": enumerate,
                                          "zip": zip, "map": map, "filter": filter}},
                 local_vars)

            # 차트 저장 여부 확인
            if Path(chart_path).exists():
                charts_created.append(chart_path)
                self._stats["charts_created"] += 1

            result = local_vars.get("result", "코드 실행 완료")
            if hasattr(result, "to_string"):
                result = result.to_string()
            elif hasattr(result, "__iter__") and not isinstance(result, str):
                result = str(list(result)[:20])
            return {"result": result, "charts": charts_created}

        except Exception as e:
            logger.warning(f"코드 실행 오류: {e}\n코드: {code[:200]}")
            return {"result": f"실행 오류: {e}", "charts": []}

    # ── 유틸리티 ─────────────────────────────────────────────────

    def _profile_to_summary(self, profile: DataProfile) -> str:
        lines = [
            f"파일: {profile.file_name}",
            f"크기: {profile.row_count:,}행 × {profile.col_count}열",
            f"컬럼: {', '.join(profile.columns[:10])}",
        ]
        if profile.missing_values:
            lines.append(f"결측치: {profile.missing_values}")
        if profile.numeric_stats:
            for col, stats in list(profile.numeric_stats.items())[:3]:
                lines.append(f"{col}: 평균={stats['mean']:.2f}, 범위=[{stats['min']:.1f}, {stats['max']:.1f}]")
        if profile.correlations:
            lines.append(f"강한 상관관계: {profile.correlations[:3]}")
        if profile.anomalies:
            lines.append(f"이상치: {profile.anomalies}")
        return "\n".join(lines)

    def _extract_data_points(self, result: Any) -> List[Dict]:
        """결과에서 숫자 데이터 포인트 추출"""
        data_points = []
        if isinstance(result, str):
            numbers = re.findall(r"(\w[\w\s]*?):\s*([\d,.]+)", result)
            for label, value in numbers[:5]:
                try:
                    data_points.append({"label": label.strip(), "value": float(value.replace(",", ""))})
                except ValueError:
                    pass
        return data_points

    def _generate_recommendations(self, question: str, result: Any, profile: Optional[DataProfile]) -> List[str]:
        recs = []
        if profile:
            if profile.missing_values:
                recs.append("결측치가 있습니다 — 대체값(평균/중앙값) 또는 행 제거를 고려하세요")
            if profile.anomalies:
                recs.append("이상치가 감지되었습니다 — 원인 파악 후 처리 여부를 결정하세요")
            if len(profile.correlations) > 0:
                recs.append("강한 상관관계 변수가 있습니다 — 다중공선성 문제를 확인하세요")
        if "trend" in question.lower() or "트렌드" in question:
            recs.append("시계열 분석을 통해 장기 트렌드를 더 자세히 파악할 수 있습니다")
        return recs[:3]

    def _error_result(self, dataset: str, question: str, error: str) -> AnalysisResult:
        return AnalysisResult(
            dataset_name=dataset,
            question=question,
            answer=f"오류: {error}",
            data_points=[],
            charts_generated=[],
            recommendations=[],
            confidence=0.0,
        )

    def list_datasets(self) -> List[Dict]:
        """로드된 데이터셋 목록"""
        result = []
        for name, df in self._loaded_datasets.items():
            result.append({
                "name": name,
                "shape": list(df.shape),
                "columns": list(df.columns[:5]),
            })
        return result

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self._stats,
                "loaded_datasets": list(self._loaded_datasets.keys()),
                "pandas_available": PANDAS_AVAILABLE,
                "matplotlib_available": MATPLOTLIB_AVAILABLE,
            }

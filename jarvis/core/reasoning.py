"""
JARVIS 고급 추론 엔진
- Chain-of-Thought (CoT) 추론
- Tree-of-Thought (ToT) 문제 해결
- ReAct (Reasoning + Acting) 패턴
- Self-Reflection 자기 개선
- 가설 생성 및 검증
"""

import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningStrategy(str, Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    SELF_REFLECTION = "self_reflection"
    SOCRATIC = "socratic"


@dataclass
class ThoughtStep:
    """추론 단계"""
    step_num: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    score: float = 0.0


@dataclass
class ReasoningResult:
    """추론 결과"""
    strategy: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    duration: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ReasoningEngine:
    """
    JARVIS 고급 추론 엔진
    인간을 뛰어넘는 분석적 사고 구현
    """

    COT_PROMPT = """다음 문제를 단계별로 생각해보세요. 각 단계를 명시적으로 보여주세요.

문제: {problem}

[단계별 사고 시작]
"""

    TOT_PROMPT = """다음 문제에 대해 여러 가지 접근법을 탐색하고 최선의 해결책을 찾으세요.

문제: {problem}

접근법 1: [첫 번째 해결책 아이디어]
접근법 2: [두 번째 해결책 아이디어]
접근법 3: [세 번째 해결책 아이디어]

평가 기준:
- 실현 가능성
- 효율성
- 완전성

최선의 접근법: [선택 및 이유]
"""

    REACT_PROMPT = """문제를 ReAct 방식으로 해결하세요: 생각 → 행동 → 관찰을 반복합니다.

문제: {problem}

형식:
생각: [현재 상황 분석]
행동: [수행할 행동]
관찰: [행동 결과]
... (반복)
최종 답변: [결론]
"""

    REFLECTION_PROMPT = """이전 응답을 비판적으로 검토하고 개선하세요.

원래 질문: {problem}
초기 응답: {initial_response}

검토 항목:
1. 정확성: 사실과 맞는가?
2. 완전성: 빠진 부분이 있는가?
3. 명확성: 이해하기 쉬운가?
4. 개선 사항: 무엇을 더 잘 할 수 있는가?

개선된 응답:
"""

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self.reasoning_history = []

    def reason(
        self,
        problem: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        context: str = "",
        max_steps: int = 5,
    ) -> ReasoningResult:
        """추론 실행"""
        start_time = time.time()
        result = ReasoningResult(strategy=strategy.value)

        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            result = self._chain_of_thought(problem, context, max_steps)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            result = self._tree_of_thought(problem, context)
        elif strategy == ReasoningStrategy.REACT:
            result = self._react(problem, context, max_steps)
        elif strategy == ReasoningStrategy.SELF_REFLECTION:
            result = self._self_reflection(problem, context)
        elif strategy == ReasoningStrategy.SOCRATIC:
            result = self._socratic(problem, context)

        result.duration = time.time() - start_time
        self.reasoning_history.append(result)
        return result

    def _chain_of_thought(self, problem: str, context: str, max_steps: int) -> ReasoningResult:
        """Chain-of-Thought 추론"""
        from jarvis.llm.manager import Message

        prompt = self.COT_PROMPT.format(problem=problem)
        if context:
            prompt = f"배경 정보:\n{context}\n\n" + prompt

        messages = [Message(role="user", content=prompt)]
        system = """당신은 단계별로 체계적으로 생각하는 AI입니다.
각 추론 단계를 명확히 보여주고, 최종 결론에 도달하세요.
각 단계 앞에 [단계 N] 을 표시하세요."""

        response = self.llm.chat(messages, system=system, max_tokens=4096)

        # 단계 파싱
        steps = self._parse_cot_steps(response.content)

        return ReasoningResult(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT.value,
            steps=steps,
            final_answer=response.content,
            confidence=0.85,
        )

    def _tree_of_thought(self, problem: str, context: str) -> ReasoningResult:
        """Tree-of-Thought 탐색"""
        from jarvis.llm.manager import Message

        prompt = self.TOT_PROMPT.format(problem=problem)
        if context:
            prompt = f"배경 정보:\n{context}\n\n" + prompt

        messages = [Message(role="user", content=prompt)]
        system = """당신은 여러 가지 해결책을 동시에 탐색하는 AI입니다.
각 접근법의 장단점을 평가하고 최선을 선택하세요."""

        response = self.llm.chat(messages, system=system, max_tokens=4096)

        return ReasoningResult(
            strategy=ReasoningStrategy.TREE_OF_THOUGHT.value,
            steps=[ThoughtStep(step_num=1, thought=response.content)],
            final_answer=response.content,
            confidence=0.90,
        )

    def _react(self, problem: str, context: str, max_steps: int) -> ReasoningResult:
        """ReAct 패턴 - 추론과 행동 교차"""
        from jarvis.llm.manager import Message

        prompt = self.REACT_PROMPT.format(problem=problem)
        if context:
            prompt = f"배경 정보:\n{context}\n\n" + prompt

        messages = [Message(role="user", content=prompt)]
        system = """당신은 생각-행동-관찰 사이클로 문제를 해결하는 AI입니다.
각 단계를 명시적으로 보여주고, 최종 답변을 도출하세요."""

        response = self.llm.chat(messages, system=system, max_tokens=4096)
        steps = self._parse_react_steps(response.content)

        return ReasoningResult(
            strategy=ReasoningStrategy.REACT.value,
            steps=steps,
            final_answer=self._extract_final_answer(response.content),
            confidence=0.88,
        )

    def _self_reflection(self, problem: str, context: str) -> ReasoningResult:
        """자기 반성 - 초기 응답 생성 후 검토 및 개선"""
        from jarvis.llm.manager import Message

        # 1단계: 초기 응답
        messages = [Message(role="user", content=problem)]
        initial = self.llm.chat(messages, max_tokens=2048)

        # 2단계: 자기 반성
        reflection_prompt = self.REFLECTION_PROMPT.format(
            problem=problem,
            initial_response=initial.content,
        )
        messages2 = [Message(role="user", content=reflection_prompt)]
        system = """당신은 자신의 응답을 비판적으로 검토하고 개선하는 AI입니다.
정직하게 약점을 찾고, 더 나은 응답을 제공하세요."""

        refined = self.llm.chat(messages2, system=system, max_tokens=4096)

        steps = [
            ThoughtStep(step_num=1, thought="초기 응답 생성", observation=initial.content[:500]),
            ThoughtStep(step_num=2, thought="자기 반성 및 개선", observation=refined.content[:500]),
        ]

        return ReasoningResult(
            strategy=ReasoningStrategy.SELF_REFLECTION.value,
            steps=steps,
            final_answer=refined.content,
            confidence=0.92,
        )

    def _socratic(self, problem: str, context: str) -> ReasoningResult:
        """소크라테스식 질문법 - 핵심 질문으로 분해"""
        from jarvis.llm.manager import Message

        # 핵심 질문 생성
        q_prompt = f"""'{problem}'을 이해하기 위해 답해야 하는 5가지 핵심 질문을 생성하세요.
각 질문은 더 깊은 이해로 이어져야 합니다."""

        messages = [Message(role="user", content=q_prompt)]
        questions_resp = self.llm.chat(messages, max_tokens=1024)

        # 각 질문에 답변
        answer_prompt = f"""다음 질문들에 순서대로 답하고, 종합적 결론을 도출하세요:

원래 문제: {problem}

핵심 질문들:
{questions_resp.content}

각 질문에 답한 후, 종합 결론을 제시하세요."""

        messages2 = [Message(role="user", content=answer_prompt)]
        answers_resp = self.llm.chat(messages2, max_tokens=4096)

        return ReasoningResult(
            strategy=ReasoningStrategy.SOCRATIC.value,
            steps=[
                ThoughtStep(step_num=1, thought="핵심 질문 도출", observation=questions_resp.content),
                ThoughtStep(step_num=2, thought="질문별 답변 및 종합", observation=answers_resp.content[:300]),
            ],
            final_answer=answers_resp.content,
            confidence=0.87,
        )

    def _parse_cot_steps(self, text: str) -> List[ThoughtStep]:
        """CoT 텍스트에서 단계 파싱"""
        steps = []
        pattern = r'\[단계\s*(\d+)\](.*?)(?=\[단계|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

        for i, (num, content) in enumerate(matches):
            steps.append(ThoughtStep(
                step_num=int(num),
                thought=content.strip()[:500],
            ))

        if not steps:
            # 줄바꿈 기반 파싱
            lines = text.split('\n')
            for i, line in enumerate(lines[:10]):
                if line.strip():
                    steps.append(ThoughtStep(step_num=i+1, thought=line.strip()))

        return steps

    def _parse_react_steps(self, text: str) -> List[ThoughtStep]:
        """ReAct 텍스트에서 단계 파싱"""
        steps = []
        thought_pattern = r'생각:\s*(.*?)(?=행동:|관찰:|최종|\Z)'
        action_pattern = r'행동:\s*(.*?)(?=관찰:|생각:|최종|\Z)'
        obs_pattern = r'관찰:\s*(.*?)(?=생각:|행동:|최종|\Z)'

        thoughts = re.findall(thought_pattern, text, re.DOTALL)
        actions = re.findall(action_pattern, text, re.DOTALL)
        observations = re.findall(obs_pattern, text, re.DOTALL)

        max_len = max(len(thoughts), len(actions), len(observations), 1)
        for i in range(max_len):
            steps.append(ThoughtStep(
                step_num=i+1,
                thought=thoughts[i].strip() if i < len(thoughts) else "",
                action=actions[i].strip() if i < len(actions) else "",
                observation=observations[i].strip() if i < len(observations) else "",
            ))

        return steps

    def _extract_final_answer(self, text: str) -> str:
        """최종 답변 추출"""
        pattern = r'최종\s*답변:\s*(.*?)$'
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return text.split('\n')[-1] if text else text

    def select_strategy(self, problem: str) -> ReasoningStrategy:
        """문제 유형에 따라 최적 전략 자동 선택"""
        problem_lower = problem.lower()

        # 단계적 계산이나 증명
        if any(kw in problem_lower for kw in ["증명", "계산", "단계", "어떻게", "how to", "step by step"]):
            return ReasoningStrategy.CHAIN_OF_THOUGHT

        # 여러 방법이 있는 문제
        if any(kw in problem_lower for kw in ["최선", "비교", "방법", "선택", "vs", "어느 것", "장단점"]):
            return ReasoningStrategy.TREE_OF_THOUGHT

        # 외부 정보 필요
        if any(kw in problem_lower for kw in ["검색", "찾아", "알아봐", "최신", "현재"]):
            return ReasoningStrategy.REACT

        # 복잡한 분석
        if any(kw in problem_lower for kw in ["왜", "분석", "이유", "설명", "why", "analyze"]):
            return ReasoningStrategy.SELF_REFLECTION

        # 깊은 이해가 필요한 개념
        if any(kw in problem_lower for kw in ["이해", "개념", "원리", "본질", "무엇", "what is"]):
            return ReasoningStrategy.SOCRATIC

        return ReasoningStrategy.CHAIN_OF_THOUGHT

    def get_history(self) -> List[Dict]:
        """추론 이력"""
        return [
            {
                "strategy": r.strategy,
                "steps": len(r.steps),
                "confidence": r.confidence,
                "duration": r.duration,
                "preview": r.final_answer[:100],
            }
            for r in self.reasoning_history[-10:]
        ]

"""
JARVIS Iteration 13: Anthropic Computer Use API Integration
Full desktop control - screenshot, mouse, keyboard, application management
"""
import asyncio
import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import anthropic

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.5
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from PIL import ImageGrab, Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ActionType(Enum):
    SCREENSHOT = "screenshot"
    CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    DRAG = "left_click_drag"
    MOVE = "mouse_move"
    CURSOR_POS = "cursor_position"


@dataclass
class ComputerAction:
    action: ActionType
    coordinate: Optional[tuple[int, int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    start_coordinate: Optional[tuple[int, int]] = None
    direction: Optional[str] = None
    amount: Optional[int] = None


@dataclass
class ActionResult:
    success: bool
    output: Any = None
    screenshot_b64: Optional[str] = None
    error: Optional[str] = None
    action_taken: Optional[str] = None


class ComputerUseEngine:
    """
    Anthropic Computer Use API - gives JARVIS full desktop control.
    Uses Claude's vision to understand screen state and take precise actions.
    """

    def __init__(self, anthropic_client: anthropic.Anthropic):
        self.client = anthropic_client
        self.model = "claude-opus-4-6"  # Computer use requires Opus
        self.action_history: list[dict] = []
        self.screen_width, self.screen_height = self._get_screen_size()
        self._running_task: Optional[asyncio.Task] = None
        print(f"[ComputerUse] Initialized - Screen: {self.screen_width}x{self.screen_height}")

    def _get_screen_size(self) -> tuple[int, int]:
        if PYAUTOGUI_AVAILABLE:
            return pyautogui.size()
        return 1920, 1080

    def capture_screenshot(self) -> Optional[str]:
        """Capture current screen as base64 PNG."""
        if PIL_AVAILABLE:
            try:
                screenshot = ImageGrab.grab()
                screenshot = screenshot.resize((1280, 800), Image.LANCZOS)
                import io
                buf = io.BytesIO()
                screenshot.save(buf, format="PNG")
                return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"[ComputerUse] Screenshot error: {e}")
        # Fallback: use pyautogui
        if PYAUTOGUI_AVAILABLE:
            try:
                import io
                shot = pyautogui.screenshot()
                shot = shot.resize((1280, 800))
                buf = io.BytesIO()
                shot.save(buf, format="PNG")
                return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"[ComputerUse] PyAutoGUI screenshot error: {e}")
        return None

    def execute_action(self, action: ComputerAction) -> ActionResult:
        """Execute a single computer action."""
        if not PYAUTOGUI_AVAILABLE:
            return ActionResult(success=False, error="pyautogui not available")

        try:
            if action.action == ActionType.SCREENSHOT:
                screenshot = self.capture_screenshot()
                return ActionResult(success=True, screenshot_b64=screenshot, action_taken="screenshot")

            elif action.action == ActionType.CLICK:
                x, y = action.coordinate
                pyautogui.click(x, y)
                return ActionResult(success=True, action_taken=f"click({x},{y})")

            elif action.action == ActionType.RIGHT_CLICK:
                x, y = action.coordinate
                pyautogui.rightClick(x, y)
                return ActionResult(success=True, action_taken=f"right_click({x},{y})")

            elif action.action == ActionType.DOUBLE_CLICK:
                x, y = action.coordinate
                pyautogui.doubleClick(x, y)
                return ActionResult(success=True, action_taken=f"double_click({x},{y})")

            elif action.action == ActionType.TYPE:
                pyautogui.typewrite(action.text, interval=0.05)
                return ActionResult(success=True, action_taken=f"type({action.text[:30]}...)")

            elif action.action == ActionType.KEY:
                pyautogui.hotkey(*action.key.split('+')) if '+' in action.key else pyautogui.press(action.key)
                return ActionResult(success=True, action_taken=f"key({action.key})")

            elif action.action == ActionType.SCROLL:
                x, y = action.coordinate
                amount = action.amount or 3
                if action.direction == "up":
                    pyautogui.scroll(amount, x=x, y=y)
                else:
                    pyautogui.scroll(-amount, x=x, y=y)
                return ActionResult(success=True, action_taken=f"scroll({action.direction},{amount})")

            elif action.action == ActionType.DRAG:
                sx, sy = action.start_coordinate
                ex, ey = action.coordinate
                pyautogui.drag(ex - sx, ey - sy, duration=0.5, button='left')
                return ActionResult(success=True, action_taken=f"drag({sx},{sy})->({ex},{ey})")

            elif action.action == ActionType.MOVE:
                x, y = action.coordinate
                pyautogui.moveTo(x, y, duration=0.3)
                return ActionResult(success=True, action_taken=f"move({x},{y})")

            elif action.action == ActionType.CURSOR_POS:
                pos = pyautogui.position()
                return ActionResult(success=True, output={"x": pos.x, "y": pos.y}, action_taken="cursor_position")

        except Exception as e:
            return ActionResult(success=False, error=str(e))

        return ActionResult(success=False, error="Unknown action")

    async def run_computer_task(self, task: str, max_steps: int = 30) -> dict:
        """
        Use Anthropic Computer Use API to autonomously complete a desktop task.
        JARVIS sees the screen and takes actions until task is complete.
        """
        print(f"[ComputerUse] Starting task: {task}")
        messages = []
        steps_taken = []
        result_log = []

        # Initial screenshot
        screenshot_b64 = self.capture_screenshot()
        if screenshot_b64:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}
                    },
                    {"type": "text", "text": f"Task: {task}\n\nThis is the current screen. Complete the task."}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": task
            })

        for step in range(max_steps):
            try:
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    tools=[{
                        "type": "computer_20241022",
                        "name": "computer",
                        "display_width_px": 1280,
                        "display_height_px": 800,
                        "display_number": 1
                    }],
                    messages=messages,
                    betas=["computer-use-2024-10-22"]
                )

                # Process response
                tool_uses = []
                text_parts = []

                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use" and block.name == "computer":
                            tool_uses.append(block)

                if text_parts:
                    result_log.append(f"JARVIS: {' '.join(text_parts)}")

                # Stop if no tool use (task complete)
                if response.stop_reason == "end_turn" or not tool_uses:
                    break

                # Add assistant response
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool use
                tool_results = []
                for tool_use in tool_uses:
                    input_data = tool_use.input
                    action_name = input_data.get("action", "")
                    coordinate = input_data.get("coordinate")
                    text = input_data.get("text")
                    key = input_data.get("key")
                    start_coord = input_data.get("start_coordinate")
                    direction = input_data.get("direction")
                    amount = input_data.get("amount")

                    # Map to ActionType
                    action_map = {
                        "screenshot": ActionType.SCREENSHOT,
                        "left_click": ActionType.CLICK,
                        "right_click": ActionType.RIGHT_CLICK,
                        "double_click": ActionType.DOUBLE_CLICK,
                        "type": ActionType.TYPE,
                        "key": ActionType.KEY,
                        "scroll": ActionType.SCROLL,
                        "left_click_drag": ActionType.DRAG,
                        "mouse_move": ActionType.MOVE,
                        "cursor_position": ActionType.CURSOR_POS,
                    }

                    action_type = action_map.get(action_name, ActionType.SCREENSHOT)
                    action = ComputerAction(
                        action=action_type,
                        coordinate=tuple(coordinate) if coordinate else None,
                        text=text,
                        key=key,
                        start_coordinate=tuple(start_coord) if start_coord else None,
                        direction=direction,
                        amount=amount
                    )

                    result = self.execute_action(action)
                    steps_taken.append({
                        "step": step + 1,
                        "action": action_name,
                        "input": input_data,
                        "success": result.success
                    })

                    if result.success:
                        if result.screenshot_b64 or action_name == "screenshot":
                            # Take fresh screenshot
                            time.sleep(0.5)
                            screenshot = self.capture_screenshot()
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": "image/png",
                                                   "data": screenshot or result.screenshot_b64}
                                    }
                                ]
                            })
                        else:
                            # After action, take screenshot to show result
                            time.sleep(0.8)
                            screenshot = self.capture_screenshot()
                            content = []
                            if screenshot:
                                content.append({
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": screenshot}
                                })
                            content.append({"type": "text", "text": f"Action completed: {result.action_taken}"})
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": content
                            })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Error: {result.error}",
                            "is_error": True
                        })

                messages.append({"role": "user", "content": tool_results})

            except anthropic.BadRequestError as e:
                print(f"[ComputerUse] API error at step {step}: {e}")
                break
            except Exception as e:
                print(f"[ComputerUse] Error at step {step}: {e}")
                break

        self.action_history.append({
            "task": task,
            "steps": len(steps_taken),
            "log": result_log,
            "completed": True
        })

        return {
            "task": task,
            "steps_taken": steps_taken,
            "total_steps": len(steps_taken),
            "log": result_log,
            "success": len(steps_taken) > 0
        }

    def open_application(self, app_name: str) -> ActionResult:
        """Open an application by name."""
        app_map = {
            "notepad": "notepad.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "vscode": "code",
            "terminal": "wt.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
        }

        exe = app_map.get(app_name.lower(), app_name)
        try:
            subprocess.Popen(exe, shell=True)
            time.sleep(2)
            return ActionResult(success=True, action_taken=f"opened:{exe}")
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def close_application(self, process_name: str) -> ActionResult:
        """Close application by process name."""
        try:
            subprocess.run(["taskkill", "/f", "/im", process_name], capture_output=True)
            return ActionResult(success=True, action_taken=f"closed:{process_name}")
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def get_status(self) -> dict:
        return {
            "screen_size": f"{self.screen_width}x{self.screen_height}",
            "pyautogui_available": PYAUTOGUI_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
            "actions_completed": len(self.action_history),
            "model": self.model
        }

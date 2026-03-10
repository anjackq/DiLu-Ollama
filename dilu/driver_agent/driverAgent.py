import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from rich import print
from typing import List

# UPDATED IMPORTS
from langchain_openai import AzureChatOpenAI, ChatOpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import OpenAICallbackHandler

from dilu.scenario.envScenario import EnvScenario

delimiter = "####"
ACTION_RECOVERY_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*(?:<[^>]+>\s*)?([0-4])\s*$", re.IGNORECASE | re.MULTILINE)
ACTION_ANYWHERE_PATTERN = re.compile(r"\b([0-4])\b")


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is not None:
                    chunks.append(str(text))
                else:
                    chunks.append(str(part))
            else:
                text = getattr(part, "text", None)
                chunks.append(str(text) if text is not None else str(part))
        return "".join(chunks)
    return str(content)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except Exception:
        return default
# ... (Keep example_message and example_answer variables as they are in original) ...
example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes, and you are currently driving in the second lane from the left. Your speed is 25.00 m/s, acceleration is 0.00 m/s^2, and lane position is 363.14 m. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `912` is driving on the same lane of you and is ahead of you. The speed of it is 23.30 m/s, acceleration is 0.00 m/s^2, and lane position is 382.33 m.
        - Vehicle `864` is driving on the lane to your right and is ahead of you. The speed of it is 21.30 m/s, acceleration is 0.00 m/s^2, and lane position is 373.74 m.
        - Vehicle `488` is driving on the lane to your left and is ahead of you. The speed of it is 23.61 $m/s$, acceleration is 0.00 $m/s^2$, and lane position is 368.75 $m$.

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-left - change lane to the left of the current lane Action_id: 0
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """)
example_answer = textwrap.dedent(f"""\
        **Step-by-Step Explanation:**
        1. **Safety Check:** The vehicle directly ahead in my lane (Vehicle 912) is only 19.19 meters away (382.33m - 363.14m), which is under the 25m safety buffer. It is also traveling slower than me (23.30 m/s vs 25.00 m/s). This presents an immediate collision risk.
        2. **Efficiency Consideration:** My current speed is 25.00 m/s, which is close to the 28 m/s target, but safety supersedes efficiency. Accelerating or maintaining speed will cause a rear-end collision.
        3. **Lane Change Feasibility:** Changing lanes is not safe. The left lane is blocked by Vehicle 488 (only 5.61m ahead), and the right lane is blocked by Vehicle 864 (only 10.6m ahead). Both are too close to attempt a safe lane change.
        4. **Conclusion:** Since I cannot safely maintain speed, accelerate, or change lanes due to surrounding traffic, I must decelerate to avoid crashing into Vehicle 912.

        **Answer:**
        Reasoning: The lead car in my lane is critically close (under 25m) and slower, and adjacent lanes are blocked, mandating immediate deceleration.
        Response to user:{delimiter} 4
        """)


class DriverAgent:
    def __init__(
            self, sce: EnvScenario,
            temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        self.decision_timeout_sec = _env_float("DILU_DECISION_TIMEOUT_SEC", 60.0)
        # For local Ollama models, invoke mode avoids long stream stalls on small models.
        default_streaming = oai_api_type != "ollama"
        self.use_streaming = _env_bool("DILU_USE_STREAMING", default_streaming)
        self.enable_checker_llm = _env_bool("DILU_ENABLE_CHECKER_LLM", True)
        max_tokens_default = 2000
        self.max_tokens = int(os.getenv("DILU_MAX_OUTPUT_TOKENS", str(max_tokens_default)))
        self.last_decision_meta = {
            "timed_out": False,
            "used_fallback": False,
            "fallback_reason": None,
            "parse_mode": "unknown",
            "checker_used": False,
            "selected_action": None,
            "decision_elapsed_sec": 0.0,
        }
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[
                    OpenAICallbackHandler()
                ],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.decision_timeout_sec,
                streaming=self.use_streaming,
            )
        elif oai_api_type == "openai":
            print("Use OpenAI API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                callbacks=[
                    OpenAICallbackHandler()
                ],
                model_name=os.getenv("OPENAI_CHAT_MODEL"),
                max_tokens=self.max_tokens,
                request_timeout=self.decision_timeout_sec,
                streaming=self.use_streaming,
            )
        # [ADD] Added support for local Ollama models
        elif oai_api_type == "ollama":
            model_name = os.getenv("OLLAMA_CHAT_MODEL")
            api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")

            print(f"Using Local Ollama API: {model_name} at {api_base}")

            # Use ChatOpenAI to talk to Ollama's OpenAI-compatible endpoint
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                openai_api_base=api_base,  # or base_url for newer langchain versions
                openai_api_key=api_key,
                max_tokens=self.max_tokens,
                request_timeout=self.decision_timeout_sec,
                streaming=self.use_streaming,
            )
        elif oai_api_type == "gemini":
            if ChatGoogleGenerativeAI is None:
                raise ImportError(
                    "Gemini support requires 'langchain-google-genai'. Install with: pip install langchain-google-genai"
                )
            model_name = os.getenv("GEMINI_CHAT_MODEL")
            api_key = os.getenv("GEMINI_API_KEY")
            if not model_name:
                raise ValueError("GEMINI_CHAT_MODEL is not configured.")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            print(f"Using Gemini API: {model_name}")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=self.max_tokens,
                timeout=self.decision_timeout_sec,
            )
        else:
            raise ValueError(f"Unknown OPENAI_API_TYPE: {oai_api_type}")

    def _run_with_timeout(self, fn, *args):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=self.decision_timeout_sec)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Decision timeout after {self.decision_timeout_sec:.1f}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _invoke_response(self, messages) -> str:
        response = self._run_with_timeout(self.llm.invoke, messages)
        return _content_to_text(getattr(response, "content", ""))

    def _stream_response(self, messages) -> str:
        def _collect_stream(msgs):
            chunks = []
            for chunk in self.llm.stream(msgs):
                chunk_text = _content_to_text(getattr(chunk, "content", ""))
                if chunk_text:
                    chunks.append(chunk_text)
                    print(chunk_text, end="", flush=True)
            return "".join(chunks)

        return self._run_with_timeout(_collect_stream, messages)

    def few_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available",
                          available_actions: str = "Not available", driving_intensions: str = "Not available",
                          fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are an autonomous driving decision module. You must strictly follow a Chain of Thought reasoning process before making a decision.

        ### DRIVE LOGIC:
        1. SAFETY: If a lead car is closer than 25m and your speed is higher, you MUST decelerate (Action_id 4).
        2. NO-UNNECESSARY-LANE-CHANGE: If the lead car in your current lane is not slower than you (or not close enough to block you), do not change lane just for preference.
        3. EFFICIENCY: Maintain a target speed of 28m/s.
        4. TRAFFIC RULE (RIGHT-HAND TRAFFIC): Prefer overtaking from the LEFT lane.
           - If blocked by slower traffic in your lane, first consider Turn-left (Action_id 0) when safe.
           - Do NOT choose Turn-right (Action_id 2) to overtake a slower vehicle in front unless there is a clear safety reason and left change is not feasible.
           - If no safe lane change exists and risk is increasing, decelerate (Action_id 4).

        You MUST format your response EXACTLY like this, using these exact headings:

        **Step-by-Step Explanation:**
        1. **Safety Check:** [Analyze distance and speed of the lead car and adjacent cars]
        2. **Efficiency Consideration:** [Analyze if you need to speed up to reach the target speed]
        3. **Lane Change Feasibility:** [Analyze if changing lanes is safe or necessary]
        4. **Conclusion:** [Summarize the best course of action]

        **Answer:**
        Reasoning: [1-sentence summary of the conclusion]
        Response to user:{delimiter} 4

        IMPORTANT:
        - Output a real integer (0-4), not a placeholder.
        - Do NOT output angle brackets like <Action_id_integer>.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )
        # print("fewshot number:", (len(messages) - 2)/2)
        start_time = time.time()
        decision_meta = {
            "timed_out": False,
            "used_fallback": False,
            "fallback_reason": None,
            "parse_mode": "direct",
            "checker_used": False,
            "selected_action": None,
            "decision_elapsed_sec": 0.0,
        }

        # NOTE: get_openai_callback might return 0 for Ollama
        # with get_openai_callback() as cb:
        # response = self.llm.invoke(messages) # invoke instead of __call__

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        try:
            if self.use_streaming:
                response_content = self._stream_response(messages)
            else:
                response_content = self._invoke_response(messages)
                print(response_content, end="", flush=True)
            print("\n")
        except TimeoutError:
            response_content = f"Decision timeout. Response to user:{delimiter} 4"
            print(f"\n[red]Decision timeout after {self.decision_timeout_sec:.1f}s. Fallback action: 4[/red]")
            decision_meta["timed_out"] = True
            decision_meta["used_fallback"] = True
            decision_meta["fallback_reason"] = "decision_timeout"
            decision_meta["parse_mode"] = "timeout_fallback"
            decision_meta["selected_action"] = 4
            decision_meta["decision_elapsed_sec"] = round(time.time() - start_time, 3)
            self.last_decision_meta = decision_meta
            few_shot_answers_store = ""
            for i in range(len(fewshot_messages)):
                few_shot_answers_store += fewshot_answers[i] + "\n---------------\n"
            print("Result:", 4)
            return 4, response_content, human_message, few_shot_answers_store

        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action.strip())
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            # Common SLM failure mode: emits placeholder line and then the real integer on the next line.
            # Recover locally before spending another LLM call on the checker.
            matches = ACTION_RECOVERY_PATTERN.findall(response_content)
            if matches:
                result = int(matches[-1])
                decision_meta["parse_mode"] = "regex_recovered"
                print(f"[yellow]Recovered action via regex parse:[/yellow] {result}")
            else:
                if not self.enable_checker_llm:
                    any_matches = ACTION_ANYWHERE_PATTERN.findall(response_content)
                    if any_matches:
                        result = int(any_matches[-1])
                        decision_meta["parse_mode"] = "loose_recovered"
                        print(f"[yellow]Recovered action from loose parse:[/yellow] {result}")
                    else:
                        result = 4
                        decision_meta["used_fallback"] = True
                        decision_meta["fallback_reason"] = "parse_fallback"
                        decision_meta["parse_mode"] = "parse_fallback"
                        print("[red]Output parse failed. Checker disabled; fallback to action 4.[/red]")
                else:
                    decision_meta["checker_used"] = True
                    print("Output is not a int number, checking the output...")
                    check_message = f"""
                    You are a output checking assistant who is responsible for checking the output of another agent.

                    The output you received is: {decision_action}

                    Your should just output the right int type of action_id, with no other characters or delimiters.
                    i.e. :
                    | Action_id | Action Description                                     |
                    |--------|--------------------------------------------------------|
                    | 0      | Turn-left: change lane to the left of the current lane |
                    | 1      | IDLE: remain in the current lane with current speed   |
                    | 2      | Turn-right: change lane to the right of the current lane|
                    | 3      | Acceleration: accelerate the vehicle                 |
                    | 4      | Deceleration: decelerate the vehicle                 |


                    You answer format would be:
                    {delimiter} <correct action_id within 0-4>
                    """
                    messages = [
                        HumanMessage(content=check_message),
                    ]
                    try:
                        check_response = self._run_with_timeout(self.llm.invoke, messages)
                        check_text = _content_to_text(getattr(check_response, "content", "")).strip()
                    except TimeoutError:
                        check_text = ""
                        decision_meta["timed_out"] = True
                        print("[yellow]Checker timed out. Applying safe fallback parse.[/yellow]")

                    tail = check_text.split(delimiter)[-1].strip() if delimiter in check_text else check_text
                    try:
                        result = int(tail)
                        if result < 0 or result > 4:
                            raise ValueError
                        decision_meta["parse_mode"] = "checker_direct"
                    except ValueError:
                        matches = ACTION_RECOVERY_PATTERN.findall(check_text)
                        if matches:
                            result = int(matches[-1])
                            decision_meta["parse_mode"] = "checker_regex_recovered"
                            print(f"[yellow]Recovered action from checker output:[/yellow] {result}")
                        else:
                            any_matches = ACTION_ANYWHERE_PATTERN.findall(check_text)
                            if any_matches:
                                result = int(any_matches[-1])
                                decision_meta["parse_mode"] = "checker_loose_recovered"
                                print(f"[yellow]Recovered action from loose parse:[/yellow] {result}")
                            else:
                                # Safety-first fallback for driving.
                                result = 4
                                decision_meta["used_fallback"] = True
                                decision_meta["fallback_reason"] = "checker_fallback"
                                decision_meta["parse_mode"] = "checker_fallback"
                                print("[red]Checker output parse failed. Falling back to safe action 4 (Deceleration).[/red]")

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                      "\n---------------\n"
        decision_meta["selected_action"] = int(result)
        decision_meta["decision_elapsed_sec"] = round(time.time() - start_time, 3)
        self.last_decision_meta = decision_meta
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store

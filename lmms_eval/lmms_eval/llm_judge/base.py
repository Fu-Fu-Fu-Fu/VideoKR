import abc
from typing import Any, Dict, List, Optional, Tuple, Union

from .protocol import Request, Response, ServerConfig
from .utils import JudgePromptBuilder, ResponseParser


class ServerInterface(abc.ABC):
    """Abstract base class for VLM judge providers."""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig(model_name="gpt-4o")

    @abc.abstractmethod
    def evaluate(self, request: Request) -> Response:
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        pass

    def prepare_messages(self, request: Request) -> List[Dict[str, Any]]:
        messages = request.messages.copy()
        if self.config.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        return messages

    def evaluate_binary(self, question: str, answer: str, prediction: str, output_format: str = "0/1", custom_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        prompt = JudgePromptBuilder.build_binary_prompt(question=question, answer=answer, prediction=prediction, output_format=output_format, custom_prompt=custom_prompt, **kwargs)
        request = Request(messages=[{"role": "user", "content": prompt}], question=question, answer=answer, prediction=prediction, config=self.config)
        response = self.evaluate(request)
        parsed_result = ResponseParser.parse_binary_response(response.content, output_format)
        return {"result": parsed_result, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

    def evaluate_comparative(
        self,
        question: str,
        response1: str,
        response2: str,
        context: Optional[str] = None,
        score_range: Tuple[int, int] = (1, 10),
        custom_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = JudgePromptBuilder.build_comparative_prompt(question=question, response1=response1, response2=response2, context=context, score_range=score_range, custom_prompt=custom_prompt, **kwargs)
        request = Request(messages=[{"role": "user", "content": prompt}], question=question, response1=response1, response2=response2, context=context, images=images, config=self.config)
        response = self.evaluate(request)
        scores = ResponseParser.parse_comparative_response(response.content)
        return {"scores": scores, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

    def evaluate_with_rubric(self, question: str, prediction: str, rubric: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        rubric_text = "\n".join([f"- {key}: {value}" for key, value in rubric.items()])
        prompt = f"""Evaluate the following response according to the given rubric.

Question: {question}

Response: {prediction}

Rubric:
{rubric_text}

Provide a JSON response with scores for each rubric item."""
        request = Request(messages=[{"role": "user", "content": prompt}], config=self.config)
        response = self.evaluate(request)
        parsed_result = ResponseParser.parse_json_response(response.content)
        return {"scores": parsed_result, "raw_response": response.content, "model": response.model_used, "prompt": prompt, "success": response.success}

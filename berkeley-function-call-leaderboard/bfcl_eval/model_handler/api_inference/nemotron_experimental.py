import json
import os

from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.api_inference.openai import OpenAIHandler
from bfcl.model_handler.utils import (
    ast_parse,
    combine_consecutive_user_prompts,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
    convert_to_function_call,
)
from openai import OpenAI


class NemotronExperimentalHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
        self.is_fc_model = True

        self.system_prompt = os.getenv("BFCL_MODEL_SYSTEM_PROMPT", "")
        self.temperature = float(os.getenv("BFCL_MODEL_TEMPERATURE", 0.6))
        self.top_p = float(os.getenv("BFCL_MODEL_TOP_P", 0.95))
        self.max_output_tokens = int(os.getenv("BFCL_MODEL_MAX_OUTPUT_TOKENS", 16384))
        self.model_name_override = os.getenv("BFCL_MODEL_NAME_OVERRIDE", None)

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        if self.system_prompt:
            message = [{"role": "system", "content": self.system_prompt}] + message

        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        if len(tools) > 0:
            return self.generate_with_backoff(
                messages=message,
                model=self.model_name if not self.model_name_override else self.model_name_override,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_output_tokens,
                tools=tools,
            )
        else:
            return self.generate_with_backoff(
                messages=message,
                model=self.model_name if not self.model_name_override else self.model_name_override,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_output_tokens,
            )

    def decode_ast(self, result, language="Python"):
        decoded_output = []
        for invoked_function in result:
            name = list(invoked_function.keys())[0]
            params = json.loads(invoked_function[name])
            decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result, language="Python"):
        return convert_to_function_call(result)

    #### Prompting methods ####

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

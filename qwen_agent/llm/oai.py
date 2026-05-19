# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
from pprint import pformat
from typing import Dict, Iterator, List, Literal, Optional, Union

import openai

from qwen_agent.utils.utils import format_as_text_message

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import BaseChatModel, ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FunctionCall, Message
from qwen_agent.log import logger


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        """Use native OpenAI-compatible tool calling instead of prompt-based function calling.

        This bypasses BaseFnCallModel's prompt-based approach and sends tools directly to the API.
        The native tool_calls in the response are already in function_call format,
        so we mark them to skip prompt-based postprocessing.
        """
        if delta_stream:
            raise NotImplementedError('delta_stream=True is not supported for function calling.')
        generate_cfg = copy.deepcopy(generate_cfg)
        for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
            if k in generate_cfg:
                del generate_cfg[k]
        # Convert functions to OpenAI tools format and pass directly to API
        tools = [{'type': 'function', 'function': f} for f in functions]
        generate_cfg['tools'] = tools
        # Set flag to skip prompt-based postprocessing
        self._native_tool_calling = True
        return self._chat(messages, stream=stream, delta_stream=False, generate_cfg=generate_cfg)

    def _postprocess_messages(self, messages, fncall_mode, generate_cfg):
        """Skip prompt-based function call postprocessing when using native tool calling.

        If any message already has function_call set (from native tool_calls),
        skip the prompt-based <tool_call> tag parsing.
        """
        if getattr(self, '_native_tool_calling', False):
            return BaseChatModel._postprocess_messages(self, messages, fncall_mode=False, generate_cfg=generate_cfg)
        return super()._postprocess_messages(messages, fncall_mode=fncall_mode, generate_cfg=generate_cfg)

    def _preprocess_messages(self, messages, lang, generate_cfg, functions=None, use_raw_api=False):
        """Skip prompt-based function call preprocessing when using native tool calling."""
        if getattr(self, '_native_tool_calling', False):
            return BaseChatModel._preprocess_messages(self, messages, lang=lang, generate_cfg=generate_cfg,
                                                      functions=functions)
        return super()._preprocess_messages(messages, lang=lang, generate_cfg=generate_cfg,
                                            functions=functions, use_raw_api=use_raw_api)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        logger.debug(f'LLM Input generate_cfg: \n{generate_cfg}')
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        reasoning_text = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                        if reasoning_text:
                            yield [
                                Message(role=ASSISTANT,
                                        content='',
                                        reasoning_content=reasoning_text)
                            ]
                        if hasattr(delta, 'content') and delta.content:
                            yield [Message(role=ASSISTANT, content=delta.content)]
            else:
                full_response = ''
                full_reasoning_content = ''
                full_tool_calls = []
                for chunk in response:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        reasoning_text = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                        if reasoning_text:
                            full_reasoning_content += reasoning_text
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                        if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            for tc in chunk.choices[0].delta.tool_calls:
                                if full_tool_calls and (not tc.id or
                                                        tc.id == full_tool_calls[-1]['extra']['function_id']):
                                    if tc.function.name:
                                        full_tool_calls[-1].function_call['name'] += tc.function.name
                                    if tc.function.arguments:
                                        full_tool_calls[-1].function_call['arguments'] += tc.function.arguments
                                else:
                                    full_tool_calls.append(
                                        Message(role=ASSISTANT,
                                                content='',
                                                function_call=FunctionCall(name=tc.function.name,
                                                                           arguments=tc.function.arguments),
                                                extra={'function_id': tc.id}))

                        res = []
                        if full_reasoning_content:
                            res.append(Message(role=ASSISTANT, content='', reasoning_content=full_reasoning_content))
                        if full_response:
                            res.append(Message(
                                role=ASSISTANT,
                                content=full_response,
                            ))
                        if full_tool_calls:
                            res += full_tool_calls
                        yield res
                logger.info(f'[OAI stream] finished: '
                            f'response_len={len(full_response)}, '
                            f'reasoning_len={len(full_reasoning_content)}, '
                            f'tool_calls={len(full_tool_calls)}, '
                            f'response_preview={repr(full_response[:200])}')
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            if hasattr(response.choices[0].message, 'reasoning_content'):
                return [
                    Message(role=ASSISTANT,
                            content=response.choices[0].message.content,
                            reasoning_content=response.choices[0].message.reasoning_content)
                ]
            else:
                return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def convert_messages_to_dicts(self, messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
        messages = [msg.model_dump() for msg in messages]
        messages = self._conv_qwen_agent_messages_to_oai(messages)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input: \n{pformat(messages, indent=2)}')
        return messages

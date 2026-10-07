"""Run in an AstrBot environment: python -m unittest discover -s tests -v."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from astrbot.core.provider.provider import Provider

from core.config import LLMConfig
from core.llm_action import LLMAction


class FixedPersonaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.persona = {
            "prompt": "Alice persona version 1",
            "_begin_dialogs_processed": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Alice here"},
            ],
        }
        self.manager = SimpleNamespace(
            get_persona_v3_by_id=Mock(return_value=self.persona),
            resolve_selected_persona=AsyncMock(
                return_value=("session", self.persona, None, False)
            ),
        )
        self.context = SimpleNamespace(
            persona_manager=self.manager,
            conversation_manager=SimpleNamespace(
                get_curr_conversation_id=AsyncMock(return_value="conversation"),
                get_conversation=AsyncMock(
                    return_value=SimpleNamespace(persona_id="session")
                ),
            ),
            get_config=Mock(return_value={}),
        )
        self.llm_cfg = LLMConfig(
            {
                "persona_id": "alice",
                "post_provider_id": "",
                "post_prompt": "Write a post",
                "comment_provider_id": "",
                "comment_prompt": "Write a comment",
                "reply_provider_id": "",
                "reply_prompt": "Write a reply",
            }
        )
        self.action = LLMAction(
            SimpleNamespace(llm=self.llm_cfg, context=self.context, client=Mock())
        )
        self.event = SimpleNamespace(unified_msg_origin="qq:FriendMessage:123")

    async def test_fixed_persona_used_without_event_and_overrides_session(self):
        for event in (None, self.event):
            prompt, dialogs = await self.action._get_persona_context(event)
            self.assertEqual(prompt, self.persona["prompt"])
            self.assertEqual(dialogs, self.persona["_begin_dialogs_processed"])
        self.manager.resolve_selected_persona.assert_not_awaited()

    async def test_persona_edit_is_seen_on_next_generation(self):
        await self.action._get_persona_context(None)
        self.manager.get_persona_v3_by_id.return_value = {
            **self.persona,
            "prompt": "Alice persona version 2",
        }
        prompt, _ = await self.action._get_persona_context(None)
        self.assertEqual(prompt, "Alice persona version 2")

    async def test_missing_fixed_persona_does_not_fall_back(self):
        self.manager.get_persona_v3_by_id.return_value = None
        with self.assertRaisesRegex(ValueError, "已停止本次生成"):
            await self.action._get_persona_context(self.event)
        self.manager.resolve_selected_persona.assert_not_awaited()

    async def test_persona_read_failure_does_not_fall_back(self):
        self.manager.get_persona_v3_by_id.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(ValueError, "已停止本次生成"):
            await self.action._get_persona_context(None)

    async def test_blank_persona_keeps_session_and_scheduler_behavior(self):
        self.llm_cfg.persona_id = "  "
        self.assertEqual(await self.action._get_persona_context(None), ("", []))
        prompt, _ = await self.action._get_persona_context(self.event)
        self.assertEqual(prompt, self.persona["prompt"])
        self.manager.resolve_selected_persona.assert_awaited_once()
        self.manager.get_persona_v3_by_id.assert_not_called()

    async def test_task_and_history_preserved_without_mutating_persona(self):
        history = [{"role": "user", "content": "Group history"}]
        prompt, dialogs = await self.action._build_request_context(
            event=None, task_prompt="Write a short comment", contexts=history
        )
        self.assertIn(self.persona["prompt"], prompt)
        self.assertIn("Write a short comment", prompt)
        self.assertEqual(dialogs[-1], history[0])
        dialogs[0]["content"] = "changed"
        self.assertEqual(self.persona["_begin_dialogs_processed"][0]["content"], "Hello")

    async def test_all_generators_pass_persona_to_provider(self):
        provider = Mock(spec=Provider)
        provider.text_chat = AsyncMock(
            return_value=SimpleNamespace(completion_text='"""Generated text"""')
        )
        self.action._get_provider = Mock(return_value=provider)
        self.action._get_msg_contexts = AsyncMock(return_value=[])
        post = SimpleNamespace(text="A sunny day", rt_con="", images=[])
        comment = SimpleNamespace(nickname="Friend", content="Nice")
        for event in (None, self.event):
            await self.action.generate_post(group_id="123", event=event)
            await self.action.generate_comment(post, event=event)
            await self.action.generate_reply(post, comment, event=event)
        self.assertEqual(provider.text_chat.await_count, 6)
        for call in provider.text_chat.await_args_list:
            self.assertIn(self.persona["prompt"], call.kwargs["system_prompt"])
            self.assertEqual(
                call.kwargs["contexts"][:2], self.persona["_begin_dialogs_processed"]
            )

    def test_existing_config_gets_empty_default(self):
        raw = {k: v for k, v in self.llm_cfg.raw_data().items() if k != "persona_id"}
        self.assertEqual(LLMConfig(raw).persona_id, "")
        self.assertEqual(raw["persona_id"], "")

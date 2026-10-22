# SPDX-License-Identifier: Apache-2.0
"""OpenAI endpoint admission, without dependencies on a private console."""
import os
from .q6_queue import Q6Queue


class Q6Middleware:
    def __init__(self, app):
        self.app = Q6Queue(app, max_running=int(os.environ.get('Q6_MAX_RUNNING','2')),
                           max_waiting=int(os.environ.get('Q6_MAX_WAITING','4')),
                           api_key=os.environ.get('VLLM_API_KEY',''))

    async def __call__(self, scope, receive, send):
        return await self.app(scope, receive, send)

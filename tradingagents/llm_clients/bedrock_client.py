from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "callbacks", "region_name", "credentials_profile_name", "max_tokens",
)


class BedrockClient(BaseLLMClient):
    """Client for Amazon Bedrock models via ChatBedrockConverse.

    Authentication uses standard AWS credential resolution: environment
    variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION),
    ~/.aws/credentials profile, or IAM instance/task roles.

    Install the required package: pip install langchain-aws
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatBedrockConverse instance."""
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as e:
            raise ImportError(
                "langchain-aws is required for Amazon Bedrock support. "
                "Install it with: pip install langchain-aws"
            ) from e

        self.warn_if_unknown_model()
        llm_kwargs = {"model_id": self.model}

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        class NormalizedChatBedrockConverse(ChatBedrockConverse):
            def invoke(self, inp, config=None, **kw):
                return normalize_content(super().invoke(inp, config, **kw))

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Bedrock."""
        return validate_model("bedrock", self.model)

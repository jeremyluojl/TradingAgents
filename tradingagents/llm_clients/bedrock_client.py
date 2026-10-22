from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "callbacks", "region_name", "credentials_profile_name", "max_tokens",
)

_DEFAULT_READ_TIMEOUT = 300  # seconds; Bedrock can be slow for long outputs


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

        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError as e:
            raise ImportError(
                "boto3 is required for Amazon Bedrock support. "
                "Install it with: pip install boto3"
            ) from e

        self.warn_if_unknown_model()

        read_timeout = self.kwargs.get("read_timeout", _DEFAULT_READ_TIMEOUT)
        boto_config = BotocoreConfig(read_timeout=read_timeout)
        region = self.kwargs.get("region_name")
        session = boto3.Session(profile_name=self.kwargs.get("credentials_profile_name"))
        bedrock_client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=boto_config,
        )

        llm_kwargs = {"model_id": self.model, "client": bedrock_client}

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

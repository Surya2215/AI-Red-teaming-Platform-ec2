"""Application configuration loaded from environment and .env files."""

import ipaddress
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# Cloud instance-metadata endpoints only - this platform's core feature is sending
# requests to operator-specified targets (including locally-hosted models like Ollama
# on localhost), so the guard deliberately does not block general private/internal
# ranges, only the well-known metadata services that leak cloud credentials.
_BLOCKED_METADATA_HOSTS = {"metadata.google.internal", "metadata.goog"}
_BLOCKED_METADATA_IPS = {"100.100.100.200"}  # Alibaba Cloud metadata
_BLOCKED_METADATA_NETWORK_V4 = ipaddress.ip_network("169.254.0.0/16")  # AWS/Azure/GCP/DO/Oracle IMDS + AWS ECS
_BLOCKED_METADATA_NETWORK_V6 = ipaddress.ip_network("fd00:ec2::/64")  # AWS IMDSv2 IPv6


def is_safe_target_url(url: str) -> bool:
    """Reject URLs that resolve to a cloud instance-metadata endpoint."""

    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in _BLOCKED_METADATA_HOSTS or host in _BLOCKED_METADATA_IPS:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True  # Regular DNS hostname, not an IP literal.
    if isinstance(ip, ipaddress.IPv4Address):
        return ip not in _BLOCKED_METADATA_NETWORK_V4
    return ip not in _BLOCKED_METADATA_NETWORK_V6


class Settings(BaseSettings):
    """Runtime settings shared by API, engine, and Streamlit UI."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Red Teaming and LLM Security Assessment Platform"
    environment: str = "local"
    database_url: str = "postgresql+asyncpg://redteam:redteam@localhost:5432/redteam"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: Path = ROOT_DIR / "logs"
    report_dir: Path = ROOT_DIR / "reports"
    # Opt-in: when unset, the API stays open (today's default). Set this on any
    # deployment reachable beyond localhost (this EC2 box binds uvicorn to 0.0.0.0)
    # to require a matching X-API-Key header on every request - see api.py's middleware.
    api_key: str = Field(default="", alias="API_KEY")
    garak_python: str = Field(default="", alias="GARAK_PYTHON")
    pyrit_python: str = Field(default="", alias="PYRIT_PYTHON")
    deepteam_python: str = Field(default="", alias="DEEPTEAM_PYTHON")
    # PyRIT stores all scan results (conversations, scores, attack outcomes) in one
    # shared SQLite "memory" database rather than per-scan report files - see
    # worker/result_parser.py::parse_pyrit_findings. Default matches PyRIT's own
    # convention on Linux (~/.local/share/dbdata/pyrit.db); override for other OSes.
    pyrit_memory_db_path: str = Field(
        default="~/.local/share/dbdata/pyrit.db", alias="PYRIT_MEMORY_DB_PATH"
    )

    # Tool Scan async pipeline (Celery worker on the Debian EC2 box).
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672//", alias="RABBITMQ_URL")
    valkey_url: str = Field(default="redis://localhost:6379/0", alias="VALKEY_URL")
    encryption_key: str = Field(default="", alias="ENCRYPTION_KEY")

    llm_provider: Literal["azure_openai", "aws_bedrock", "ollama", "openai", "huggingface", "anthropic"] = Field(
        default="azure_openai",
        alias="LLM_PROVIDER",
    )

    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: SecretStr | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-12-01-preview", alias="AZURE_OPENAI_API_VERSION")

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-latest", alias="ANTHROPIC_MODEL")
    anthropic_base_url: str = Field(default="https://api.anthropic.com/v1", alias="ANTHROPIC_BASE_URL")
    anthropic_version: str = Field(default="2023-06-01", alias="ANTHROPIC_VERSION")

    huggingface_api_key: SecretStr | None = Field(default=None, alias="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3", alias="HUGGINGFACE_MODEL")
    huggingface_base_url: str = Field(default="https://api-inference.huggingface.co/models", alias="HUGGINGFACE_BASE_URL")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")

    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_bedrock_model_id: str = Field(default="anthropic.claude-3-haiku-20240307-v1:0", alias="AWS_BEDROCK_MODEL_ID")

    default_temperature: float = 0.2
    default_timeout_seconds: float = 30.0
    default_retry_count: int = 2
    safe_prompt_log_chars: int = 600

    # Production user auth/RBAC/audit trail (core/auth.py) - opt-in, same philosophy
    # as api_key above: unset means today's behavior (no login required) is unchanged,
    # so upgrading the platform never locks out an existing deployment mid-flight.
    # Set AUTH_ENABLED=true for any deployment with more than one human operator -
    # it adds per-user identity (for RBAC and the audit log) on top of, not instead
    # of, the existing X-API-Key header (still honored for CI/automation clients).
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    # HMAC-signs session JWTs (core/auth.py's encode/decode_access_token) - required
    # once auth_enabled is true; generate with:
    # python -c "import secrets; print(secrets.token_urlsafe(48))"
    jwt_secret: str = Field(default="", alias="JWT_SECRET")
    access_token_ttl_minutes: int = Field(default=480, alias="ACCESS_TOKEN_TTL_MINUTES")
    # Bootstrap admin, created once on startup if the users table is empty - the only
    # way to get a first login on a fresh deployment (there is no public signup).
    initial_admin_email: str = Field(default="", alias="INITIAL_ADMIN_EMAIL")
    initial_admin_password: str = Field(default="", alias="INITIAL_ADMIN_PASSWORD")

    @property
    def azure_ready(self) -> bool:
        """Return whether Azure OpenAI credentials are configured."""

        return bool(self.azure_openai_endpoint and self.azure_openai_api_key and self.azure_openai_deployment)

    @property
    def openai_ready(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)

    @property
    def anthropic_ready(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_model)

    @property
    def huggingface_ready(self) -> bool:
        return bool(self.huggingface_api_key and self.huggingface_model)

    @property
    def ollama_ready(self) -> bool:
        return bool(self.ollama_base_url and self.ollama_model)

    @property
    def aws_bedrock_ready(self) -> bool:
        return bool(self.aws_region and self.aws_bedrock_model_id)

    @property
    def llm_ready(self) -> bool:
        readiness = {
            "azure_openai": self.azure_ready,
            "openai": self.openai_ready,
            "anthropic": self.anthropic_ready,
            "huggingface": self.huggingface_ready,
            "ollama": self.ollama_ready,
            "aws_bedrock": self.aws_bedrock_ready,
        }
        return readiness.get(self.llm_provider, False)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    return settings


def enforce_production_auth_guard(settings: Settings) -> None:
    """Fail fast at startup if a production deployment forgot to turn on any
    access control at all.

    Both `api_key` and `auth_enabled` are opt-in (see their Field comments above)
    so that upgrading the platform never locks out an existing deployment
    mid-flight - but that same opt-in default means a fresh `environment=production`
    deployment can silently boot with a completely open API (no X-API-Key, no
    login) if the operator forgets to set either one. This is a standalone,
    narrowly-scoped check (not baked into get_settings()/Settings itself) so it's
    easy to call from wherever an app's startup path lives - here that's
    core/api.py's `@app.on_event("startup")` - and just as easy to port verbatim
    to a sibling deployment of this platform that wants the same guard.

    Raises RuntimeError (meant to abort startup, not be caught) when
    settings.environment == "production" and neither settings.api_key nor
    settings.auth_enabled is set - or when auth_enabled is set but jwt_secret
    isn't (every login would mint tokens signed with an empty secret). A no-op
    for every other environment value.
    """

    if settings.environment.strip().lower() != "production":
        return
    if not settings.api_key and not settings.auth_enabled:
        raise RuntimeError(
            "Refusing to start: environment=production but neither API_KEY nor "
            "AUTH_ENABLED is set. This would leave every endpoint on this "
            "deployment reachable with no authentication at all. Set API_KEY "
            "(shared-secret X-API-Key header) and/or AUTH_ENABLED=true (per-user "
            "login) before starting in production."
        )
    if settings.auth_enabled and not settings.jwt_secret:
        raise RuntimeError(
            "Refusing to start: AUTH_ENABLED is true but JWT_SECRET is not set - "
            "every login would mint tokens signed with an empty secret. Set "
            "JWT_SECRET (see core/auth.py's docstring for how to generate one)."
        )


# openai_patch.py - Patch to resolve proxy errors in the OpenAI library
# Scope strictly limited to OpenAI namespace
import logging
import os
import inspect
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_patch")

logger.info("Applying restricted patch to OpenAI library...")

# Remove proxy environment variables
for env_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if env_var in os.environ:
        logger.info(f"Removing environment variable {env_var}")
        del os.environ[env_var]

# Patch only the __init__ method of specific classes
def patch_init(cls, name):
    if not cls.__module__.startswith('openai'):
        return False  # Skip if not part of the OpenAI package

    try:
        original_init = cls.__init__

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            # Remove 'proxies' parameter
            if 'proxies' in kwargs:
                logger.info(f"Removing 'proxies' argument from {name}")
                del kwargs['proxies']

            # Call the original initializer
            original_init(self, *args, **kwargs)

        cls.__init__ = patched_init
        return True
    except (AttributeError, TypeError) as e:
        logger.warning(f"Failed to patch {name}: {e}")
        return False

# Apply patch to specific modules only
def patch_specific_modules():
    """
    Directly patch selected modules in the OpenAI library.
    """
    try:
        # 1. Patch OpenAI clients
        import openai

        # Patch OpenAI class
        if hasattr(openai, 'OpenAI'):
            if patch_init(openai.OpenAI, "openai.OpenAI"):
                logger.info("Patched openai.OpenAI")

        # Patch AzureOpenAI class
        if hasattr(openai, 'AzureOpenAI'):
            if patch_init(openai.AzureOpenAI, "openai.AzureOpenAI"):
                logger.info("Patched openai.AzureOpenAI")

        # 2. Patch base clients
        try:
            from openai._base_client import SyncHttpxClientWrapper, BaseClient

            if patch_init(SyncHttpxClientWrapper, "SyncHttpxClientWrapper"):
                logger.info("Patched SyncHttpxClientWrapper")

            if patch_init(BaseClient, "BaseClient"):
                logger.info("Patched BaseClient")
        except ImportError:
            logger.warning("Failed to import base client classes")

        # 3. Patch httpx Clients (only when used inside OpenAI)
        try:
            import httpx

            original_httpx_init = httpx.Client.__init__

            @wraps(original_httpx_init)
            def patched_httpx_init(self, *args, **kwargs):
                if 'proxies' in kwargs and any('openai' in frame.filename for frame in inspect.stack()):
                    logger.info("Removing 'proxies' from httpx.Client when called by OpenAI")
                    del kwargs['proxies']

                original_httpx_init(self, *args, **kwargs)

            httpx.Client.__init__ = patched_httpx_init
            logger.info("Patched httpx.Client (only if called by OpenAI)")

            # Also patch AsyncClient similarly
            if hasattr(httpx, 'AsyncClient'):
                original_async_init = httpx.AsyncClient.__init__

                @wraps(original_async_init)
                def patched_async_init(self, *args, **kwargs):
                    if 'proxies' in kwargs and any('openai' in frame.filename for frame in inspect.stack()):
                        logger.info("Removing 'proxies' from httpx.AsyncClient when called by OpenAI")
                        del kwargs['proxies']

                    original_async_init(self, *args, **kwargs)

                httpx.AsyncClient.__init__ = patched_async_init
                logger.info("Patched httpx.AsyncClient (only if called by OpenAI)")
        except ImportError:
            logger.warning("Failed to import httpx module")

        # 4. Patch only key API resource classes
        try:
            # Patch Chat API
            if hasattr(openai.resources, 'chat'):
                try:
                    from openai.resources.chat import Completions
                    if patch_init(Completions, "openai.resources.chat.Completions"):
                        logger.info("Patched openai.resources.chat.Completions")
                except ImportError:
                    pass

            # Patch Embeddings API
            if hasattr(openai.resources, 'embeddings'):
                try:
                    from openai.resources.embeddings import Embeddings
                    if patch_init(Embeddings, "openai.resources.embeddings.Embeddings"):
                        logger.info("Patched openai.resources.embeddings.Embeddings")
                except ImportError:
                    pass
        except AttributeError:
            logger.warning("Failed to access resource classes")

    except ImportError as e:
        logger.error(f"Failed to import OpenAI module: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while patching: {e}")

# Run patch
patch_specific_modules()

logger.info("Finished patching OpenAI library. Scope restricted to avoid side effects on other libraries.")

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# 假设这些是您的项目结构中的正确导入
from lightrag import LightRAG
from lightrag.api.config import global_args, update_uvicorn_mode_config  # 使用全局参数
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes_with_multi_user,
)
from lightrag.api.routers.graph_routes import create_graph_routes_with_multi_user
from lightrag.api.routers.query_routes import create_query_routes_with_multi_user
from lightrag.api.utils_api import check_env_file, display_splash_screen
from lightrag.kg.shared_storage import initialize_pipeline_status, initialize_share_data
from lightrag.utils import EmbeddingFunc, setup_logger

load_dotenv(dotenv_path=".env", override=False)


# --- 模型和嵌入函数的动态导入 ---
# 为了保持代码的整洁，我们将动态导入所需的模块
# 这种方式可以在不修改代码的情况下，通过启动参数切换模型
SUPPORTED_BINDINGS = {
    # "lollms"          : ("lightrag.llm.lollms", ["lollms_model_complete", "lollms_embed"]),
    # "ollama"          : ("lightrag.llm.ollama", ["ollama_model_complete", "ollama_embed"]),
    "openai": ("lightrag.llm.openai", ["openai_complete_if_cache", "openai_embed"]),
    # "azure_openai"    : ("lightrag.llm.azure_openai", ["azure_openai_complete_if_cache", "azure_openai_embed"]),
    "langchain_gemini": (
        "lightrag.llm.langchain_gemini",
        ["langchain_gemini_complete", None],
    ),  # Gemini没有内置embedding
    # "hf-st"           : ("lightrag.llm.hf", [None, "hf_st_embed"]),
}


def import_from_string(path: str):
    """从字符串路径导入模块或函数。"""
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = __import__(module_path, fromlist=[obj_name])
    return getattr(module, obj_name)


# --- 全局配置 ---
# load_dotenv 会加载 .env 文件中的环境变量
setup_logger("lightrag")
log = logging.getLogger(__name__)

# 这个字典将作为我们的“工厂蓝图”，在服务器启动时填充
rag_factory_config = {}


# --- FastAPI 生命周期事件 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    args = global_args

    log.info("Server starting up...")
    log.info(f"Loading configuration with arguments: {args}")
    initialize_share_data()
    log.info("✅ Core shared data structures initialized.")
    # --- 1. 设置 LLM 模型函数 ---
    llm_func = None
    if args.llm_binding in SUPPORTED_BINDINGS:
        module_path, [func_name, *_] = SUPPORTED_BINDINGS[args.llm_binding]
        full_path = f"{module_path}.{func_name}"

        if args.llm_binding == "openai":
            openai_complete_if_cache = import_from_string(full_path)

            async def openai_wrapper(prompt, **kwargs):
                return await openai_complete_if_cache(
                    args.llm_model,
                    prompt,
                    base_url=args.llm_binding_host,
                    api_key=args.llm_binding_api_key,
                    **kwargs,
                )

            llm_func = openai_wrapper
        elif args.llm_binding == "langchain_gemini":
            log.info("Using langchain_gemini LLM binding.")
            llm_func = import_from_string(full_path)
        else:
            llm_func = import_from_string(full_path)
    else:
        raise ValueError(f"Unsupported LLM binding: {args.llm_binding}")

    # --- 2. 设置 Embedding 函数 ---
    embedding_func = None
    if args.embedding_binding == "openai":
        module_path, [_, func_name] = SUPPORTED_BINDINGS["openai"]
        embed_func_raw = import_from_string(f"{module_path}.{func_name}")

        def openai_embed_wrapper(texts: list[str]):
            log.debug(f"Using OpenAI embedding with model: {args.embedding_model}")
            return embed_func_raw(
                texts,
                model=args.embedding_model,
                base_url=args.embedding_binding_host,
                api_key=args.embedding_binding_api_key,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=args.embedding_dim,
            max_token_size=args.max_embed_tokens,
            func=openai_embed_wrapper,
        )
    elif args.embedding_binding in SUPPORTED_BINDINGS:
        module_path, [_, func_name] = SUPPORTED_BINDINGS[args.embedding_binding]
        if func_name:
            embed_func_raw = import_from_string(f"{module_path}.{func_name}")

            def embed_wrapper(texts: list[str]):
                return embed_func_raw(texts, model_name=args.embedding_model)

            embedding_func = EmbeddingFunc(
                embedding_dim=args.embedding_dim,
                max_token_size=args.max_embed_tokens,
                func=embed_wrapper,
            )
        else:
            raise ValueError(
                f"Embedding function not supported or found for binding: {args.embedding_binding}"
            )
    else:
        raise ValueError(f"Unsupported Embedding binding: {args.embedding_binding}")

    rag_factory_config = {
        "working_dir": args.working_dir,
        "llm_model_func": llm_func,
        "llm_model_name": args.llm_model,
        "llm_model_max_async": args.max_async,
        "llm_model_max_token_size": args.max_tokens,
        "chunk_token_size": int(args.chunk_size),
        "chunk_overlap_token_size": int(args.chunk_overlap_size),
        "llm_model_kwargs": {
            "host": args.llm_binding_host,
            "timeout": args.timeout,
            "options": {"num_ctx": args.max_tokens},
            "api_key": args.llm_binding_api_key,
        }
        if args.llm_binding in ["lollms", "ollama"]
        else {"timeout": args.timeout}
        if args.llm_binding == "azure_openai"
        else {},
        "embedding_func": embedding_func,
        "kv_storage": args.kv_storage,
        "graph_storage": args.graph_storage,
        "vector_storage": args.vector_storage,
        "doc_status_storage": args.doc_status_storage,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": args.cosine_threshold
        },
        "enable_llm_cache_for_entity_extract": args.enable_llm_cache_for_extract,
        "enable_llm_cache": args.enable_llm_cache,
        "auto_manage_storages_states": False,
        "max_parallel_insert": args.max_parallel_insert,
        "addon_params": {"language": args.summary_language},
    }

    app.state.rag_factory_config = rag_factory_config
    log.info("✅ LightRAG factory configuration loaded into app.state.")

    await initialize_pipeline_status()
    log.info("✅ Global pipeline status initialized (shared across all users).")

    temp_rag_for_init = LightRAG(**rag_factory_config)
    await temp_rag_for_init.initialize_storages()

    yield

    await temp_rag_for_init.finalize_storages()
    log.info("Server shutting down and resources cleaned up.")


# --- FastAPI 应用实例 ---
app = FastAPI(
    title="LightRAG Multi-Tenant API",
    description="支持多用户的 LightRAG API 服务",
    version="1.0.0-multitenant",
    lifespan=lifespan,
)


# --- 路由注册 ---
# 初始化文档管理器 (这是无状态的，可以全局共享)
doc_manager = DocumentManager(global_args.input_dir)

# 注意这里的变化：我们不再传递一个固定的 `rag` 对象。
# 路由创建函数现在将使用我们新的 `get_rag_for_user` 依赖。
# 我们也不再向路由创建函数传递 api_key，因为认证已经解耦。
app.include_router(create_document_routes_with_multi_user(doc_manager), prefix="/api")
app.include_router(
    create_query_routes_with_multi_user(global_args.top_k), prefix="/api"
)
app.include_router(create_graph_routes_with_multi_user(), prefix="/api")

# --- 静态文件和根路径 ---
static_dir = Path(__file__).parent / "webui"
app.mount("/webui", StaticFiles(directory=static_dir, html=True), name="webui")


@app.get("/")
async def root():
    return RedirectResponse(url="/webui")


@app.get("/health")
async def health():
    """Health check endpoint for container orchestration"""
    try:
        # Simple health check - verify app state is available
        if hasattr(app.state, "rag_factory_config"):
            return {
                "status": "healthy",
                "server_type": "multiuser",
                "version": "1.0.0-multitenant",
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            }
        else:
            return {
                "status": "starting",
                "server_type": "multiuser",
                "message": "Application is still initializing",
            }
    except Exception as e:
        return {"status": "unhealthy", "server_type": "multiuser", "error": str(e)}


def main():
    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    # Start Uvicorn in single process mode
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": global_args.host,
        "port": global_args.port,
        "log_config": None,  # Disable default config
    }

    if global_args.ssl:
        uvicorn_config.update({
            "ssl_certfile": global_args.ssl_certfile,
            "ssl_keyfile": global_args.ssl_keyfile,
        })

    print(
        f"Starting Uvicorn server in single-process mode on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()

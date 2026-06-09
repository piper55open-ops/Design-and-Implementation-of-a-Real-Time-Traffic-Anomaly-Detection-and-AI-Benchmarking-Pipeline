import uvicorn

if __name__ == "__main__":
    print("🚀 正在启动 FastAPI 网关服务器 (稳定压测模式)...")
    # 💡 核心修复：关掉 reload，彻底避开 Windows 的多进程底层 Bug
    uvicorn.run("app.api_gateway:app", host="127.0.0.1", port=8000, reload=False)
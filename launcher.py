#!/usr/bin/env python3
"""
Braind Project Launcher
Single command to start all services: vLLM LLM, vLLM Embed, Braind Server
"""
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from core.config import settings


class Color:
    """ANSI color codes"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class ServiceManager:
    """Manage multiple background services"""
    
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.project_root = Path(__file__).parent
        self.venv_python = self.project_root / ".venv" / "bin" / "python3"
        
    def log(self, service: str, message: str, level: str = "info"):
        """Colored logging"""
        colors = {
            "info": Color.BLUE,
            "success": Color.GREEN,
            "warning": Color.YELLOW,
            "error": Color.RED,
        }
        color = colors.get(level, Color.RESET)
        timestamp = time.strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] [{service.upper()}]{Color.RESET} {message}")
    
    def start_service(
        self, 
        name: str, 
        command: list[str], 
        cwd: Path | None = None,
        env: dict | None = None
    ):
        """Start a background service"""
        self.log(name, f"Starting: {' '.join(command)}")
        
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd or self.project_root,
                env=process_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Create new process group
            )
            self.processes[name] = process
            self.log(name, f"Started (PID: {process.pid})", "success")
            return True
        except Exception as e:
            self.log(name, f"Failed to start: {e}", "error")
            return False
    
    async def wait_for_health(
        self, 
        name: str, 
        url: str, 
        timeout: int = 120,
        check_interval: int = 2
    ):
        """Wait for service to be healthy"""
        self.log(name, f"Waiting for health check: {url}")
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        elapsed = int(time.time() - start_time)
                        self.log(name, f"Ready! (took {elapsed}s)", "success")
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                
                await asyncio.sleep(check_interval)
        
        self.log(name, f"Health check timeout after {timeout}s", "error")
        return False
    
    def stop_service(self, name: str):
        """Stop a service gracefully"""
        if name not in self.processes:
            return
        
        process = self.processes[name]
        self.log(name, "Stopping...")
        
        try:
            process.terminate()
            process.wait(timeout=10)
            self.log(name, "Stopped", "success")
        except subprocess.TimeoutExpired:
            self.log(name, "Force killing...", "warning")
            process.kill()
            process.wait()
        except Exception as e:
            self.log(name, f"Error stopping: {e}", "error")
        finally:
            del self.processes[name]
    
    def stop_all(self):
        """Stop all services"""
        self.log("manager", "Stopping all services...")
        for name in list(self.processes.keys()):
            self.stop_service(name)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        self.log("setup", "Checking prerequisites...")
        
        # Check venv exists
        if not self.venv_python.exists():
            self.log("setup", ".venv not found! Run: python3 -m venv .venv", "error")
            return False
        
        # Check PostgreSQL
        try:
            result = subprocess.run(
                ["psql", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                self.log("setup", "PostgreSQL not found!", "error")
                return False
        except Exception:
            self.log("setup", "PostgreSQL not found!", "error")
            return False
        
        self.log("setup", "Prerequisites OK", "success")
        return True


async def main():
    """Main entry point"""
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}🧠 Braind Project Launcher{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}\n")
    
    manager = ServiceManager()
    
    # Check prerequisites
    if not manager.check_prerequisites():
        sys.exit(1)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n{Color.YELLOW}Received interrupt signal{Color.RESET}")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if we should use mock vLLM (for testing)
    use_mock = "--mock" in sys.argv or "--dev" in sys.argv
    
    try:
        if use_mock:
            manager.log("launcher", "Using MOCK vLLM servers for testing", "warning")
            
            # Start mock vLLM servers
            manager.start_service(
                "vllm-embed",
                [
                    str(manager.venv_python),
                    "mock_vllm.py",
                    "8002",
                    settings.embed_model
                ]
            )

            manager.start_service(
                "vllm-llm",
                [
                    str(manager.venv_python),
                    "mock_vllm.py",
                    "8001",
                    settings.llm_model
                ]
            )
            
            # Mock servers start fast
            await asyncio.sleep(3)
        else:
            # 1. Start vLLM Embedding Server (own GPU so embed batches never
            #    contend with chat inference).
            manager.start_service(
                "vllm-embed",
                [
                    str(manager.venv_python),
                    "-m", "vllm.entrypoints.openai.api_server",
                    "--model", settings.embed_model,
                    "--host", "0.0.0.0",
                    "--port", "8002",
                    "--dtype", "auto",
                    "--runner", "pooling",
                    "--max-model-len", str(settings.vllm_embed_max_model_len),
                    "--gpu-memory-utilization", str(settings.vllm_embed_gpu_mem_util),
                ],
                env={"CUDA_VISIBLE_DEVICES": str(settings.vllm_embed_gpu)},
            )

            # 2. Start vLLM LLM Server. Tool-calling flags are mandatory: the
            #    agent loop (core/agent/loop.py) depends on native tool_calls.
            manager.start_service(
                "vllm-llm",
                [
                    str(manager.venv_python),
                    "-m", "vllm.entrypoints.openai.api_server",
                    "--model", settings.llm_model,
                    "--host", "0.0.0.0",
                    "--port", "8001",
                    "--dtype", "auto",
                    "--max-model-len", str(settings.vllm_llm_max_model_len),
                    "--gpu-memory-utilization", str(settings.vllm_llm_gpu_mem_util),
                    "--enable-auto-tool-choice",
                    "--tool-call-parser", settings.vllm_tool_call_parser,
                    "--disable-log-stats",
                ],
                env={"CUDA_VISIBLE_DEVICES": str(settings.vllm_llm_gpu)},
            )
            
            # Wait for vLLM services to load models
            manager.log("vllm", "Waiting for vLLM services to load models (90s)...")
            await asyncio.sleep(90)
        
        # Health checks
        embed_ready = await manager.wait_for_health(
            "vllm-embed",
            "http://localhost:8002/v1/models",
            timeout=30
        )
        
        llm_ready = await manager.wait_for_health(
            "vllm-llm", 
            "http://localhost:8001/v1/models",
            timeout=30
        )
        
        if not (embed_ready and llm_ready):
            manager.log("vllm", "vLLM services failed health checks", "error")
            manager.log("vllm", "Continuing anyway - services may still be loading", "warning")
        
        # 3. Start Braind Server
        manager.start_service(
            "braind",
            [
                str(manager.venv_python),
                "-m", "uvicorn",
                "api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
            ]
        )
        
        # Wait for Braind server
        await asyncio.sleep(5)
        braind_ready = await manager.wait_for_health(
            "braind",
            "http://localhost:8000/api/agents/list",
            timeout=30
        )
        
        if braind_ready:
            print(f"\n{Color.BOLD}{Color.GREEN}{'='*60}{Color.RESET}")
            print(f"{Color.BOLD}{Color.GREEN}✅ All services started successfully!{Color.RESET}")
            print(f"{Color.BOLD}{Color.GREEN}{'='*60}{Color.RESET}\n")
            print(f"{Color.BLUE}📊 Braind Office:{Color.RESET} http://localhost:8000/office")
            print(f"{Color.BLUE}🔗 API Docs:{Color.RESET} http://localhost:8000/docs")
            print(f"{Color.BLUE}🤖 vLLM LLM:{Color.RESET} http://localhost:8001/v1/models")
            print(f"{Color.BLUE}🔤 vLLM Embed:{Color.RESET} http://localhost:8002/v1/models")
            print(f"\n{Color.YELLOW}Press Ctrl+C to stop all services{Color.RESET}\n")
        else:
            manager.log("braind", "Server failed to start", "error")
            manager.stop_all()
            sys.exit(1)
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
            # Check if any process died
            for name, process in list(manager.processes.items()):
                if process.poll() is not None:
                    manager.log(name, f"Process died (exit code: {process.returncode})", "error")
                    manager.stop_all()
                    sys.exit(1)
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        manager.log("launcher", f"Error: {e}", "error")
        manager.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

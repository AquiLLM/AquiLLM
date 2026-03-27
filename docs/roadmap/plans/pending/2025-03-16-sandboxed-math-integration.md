# Sandboxed Math Integration (Python, Eigen, OpenBLAS) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable AquiLLM chat to perform advanced mathematics (numeric, linear algebra, symbolic) by executing user/LLM-requested code in a sandboxed environment using Python (NumPy/SciPy with OpenBLAS) and optional Eigen, without exposing the host or main app to arbitrary code execution. The design must also support adding **skills** and **MCP (Model Context Protocol) servers** later, so the math sandbox is the first instance of a consistent extensibility pattern.

**Architecture:** A dedicated **math-sandbox** service runs in its own Docker container with strict resource and security limits. The Django app invokes it via HTTP (or a small RPC layer). The LLM gets a new tool (e.g. `evaluate_math` or `run_python_math`) that sends code or expressions to the sandbox and returns structured results. Eigen is integrated either as a Python-accessible C++ helper inside the same container (pybind11 or subprocess to a small binary) or as a separate microservice; OpenBLAS is used as the BLAS backend for NumPy/SciPy in the sandbox image. An **extensibility layer** is introduced so that tools can come from multiple sources (built-in, math sandbox, future MCP servers, future skills); the math sandbox is implemented as the first such external tool provider, with a clear contract and registration point that MCP and skills will use later.

**Tech Stack:** Docker, Python 3.12, NumPy, SciPy, OpenBLAS (via numpy/scipy build), optional Eigen (C++/pybind11 or standalone binary), timeout/RLIMIT-based sandboxing, optional nsjail/gVisor for stronger isolation. Future: MCP SDK/client, skills loader (config/DB or directory).

---

## 1. Scope and Constraints

- **Sandbox isolation:** No network egress, no filesystem write outside a temp dir, CPU/memory/time limits, no privileged syscalls.
- **Eigen:** Header-only C++; use for performance-critical linear algebra when NumPy is insufficient or when the LLM/user explicitly needs Eigen-style APIs. Options: (A) pybind11 wrapper in sandbox image, (B) small C++ HTTP/stdio helper in same container.
- **OpenBLAS:** Provide NumPy/SciPy built/linked against OpenBLAS in the sandbox image for fast BLAS/LAPACK.
- **Tool contract:** Input: code snippet or expression string; output: `ToolResultDict` (`result`, `exception`, optional `files`) so existing LLM tool handling in `chat/consumers.py` and `llm.py` works unchanged.
- **Extensibility:** The math-sandbox integration must be implemented so that (1) adding new tools from other sources (MCP, skills) does not require scattering conditionals across the codebase, and (2) a single registration path exists for “external” tools (math, MCP, skills) that can be extended later.

---

## 2. High-Level Components

| Component | Purpose |
|-----------|--------|
| **math-sandbox service** | Container that runs Python in a restricted environment (timeout, memory cap, no network). Exposes a single endpoint, e.g. `POST /eval` with `{"code": "..."}` or `{"expression": "..."}`. |
| **Sandbox runtime** | Python process with allowed imports (numpy, scipy, math, etc.); forbidden: os.system, subprocess, socket, open outside temp, etc. Optionally run under nsjail or similar for defense in depth. |
| **Eigen integration** | Phase 2: either a pybind11 module loaded by the sandbox Python, or a small C++ binary that reads JSON from stdin and writes JSON to stdout (e.g. matrix ops). Same container as math-sandbox. |
| **OpenBLAS** | Installed in the sandbox Docker image; NumPy/SciPy built or configured to use OpenBLAS (e.g. via `numpy` wheel that links OpenBLAS or via `apt install libopenblas-dev` and building numpy from source in image; or use a pre-built scientific stack image). |
| **Django client** | Sync or async HTTP client in the web app that calls the math-sandbox service. Wrapped in an `@llm_tool` in `chat/consumers.py`. |
| **Configuration** | `.env` and `docker-compose` entries for math-sandbox URL, timeouts, enable/disable feature flag. |
| **External tool registry (extensibility)** | Single place (e.g. a list or registry in `chat/consumers.py` or a small module `aquillm/tools/`) where built-in tools, math-sandbox tool(s), and later MCP/skill tools are collected and passed to the conversation. Ensures one pattern for “add a tool from an external source.” |

---

## 2b. Extensibility: Skills and MCP (Later)

The plan must allow adding **skills** and **MCP servers** later without reworking the math integration. Below is the target model; implementation is Phase 3 (after math sandbox and OpenBLAS/Eigen).

### Skills

- **Definition:** A skill is a named bundle of capabilities: optional extra tools (each an `LLMTool`), optional system-prompt snippet, and optional configuration (e.g. when to enable the skill).
- **Discovery:** Skills are loaded from a configured source: e.g. a directory (`AQUILLM_SKILLS_DIR`), a DB table, or a config file listing skill IDs and paths. Each skill is a Python module or package that exports a function like `get_tools(user, context) -> list[LLMTool]` and optionally `get_system_prompt_extra() -> str`.
- **Registration:** At chat connect time, the consumer (or a dedicated loader) calls each enabled skill’s `get_tools(...)` and appends the returned tools to the same list used for document tools and math. System prompt can be augmented with `get_system_prompt_extra()` if present.
- **Math sandbox fit:** The math sandbox is *not* a “skill” in this sense; it is an external service that exposes one (or more) tools. The *pattern* of “append tools from an external provider” is shared: math adds tools from `math_sandbox_client`, skills add tools from skill modules, MCP adds tools from MCP servers.

### MCP (Model Context Protocol) servers

- **Definition:** MCP servers expose tools (and optionally resources and prompts) over stdio or HTTP. AquiLLM needs an **MCP client** that connects to configured servers, fetches tool definitions, and turns them into `LLMTool` instances that, when called, send requests to the MCP server and map responses to `ToolResultDict`.
- **Configuration:** `.env` (or settings) lists MCP servers, e.g. `MCP_SERVERS=math_mcp,weather_mcp` with per-server config (command or URL, env, timeouts). Optional: `MCP_ENABLED=1`.
- **Tool mapping:** Each MCP tool is represented as an `LLMTool`: `llm_definition` comes from the MCP tool schema (name, description, parameters); `_function` is a wrapper that calls the MCP client to invoke the tool and converts the MCP result to `ToolResultDict` (`result` or `exception`).
- **Registration:** Same as math and skills: at connect time, the MCP client discovers tools from all enabled MCP servers and appends them to the conversation’s tool list. No special branching in the consumer beyond “add all tools from registry / external providers.”

### Shared contract for external tools

All external tools (math sandbox, MCP, skills) must:

- Expose one or more `LLMTool` instances (same type as `vector_search`, `document_ids`, etc.).
- Return `ToolResultDict` from the underlying function (`result`, `exception`, optional `files`).
- Be registered in one place so `ChatConsumer` builds `self.tools` as: **built-in doc tools** + **math tools (if enabled)** + **skill tools (if any)** + **MCP tools (if enabled)** + **optional debug tools**.

### Implementation implications for this plan

- **Task 3 (Django client and LLM tool):** Implement the math tool so it is added via a single “external tools” collection point. For example: introduce `aquillm/aquillm/tools/external.py` (or a function in `consumers.py`) that returns `list[LLMTool]` for all enabled external providers; currently it returns `[get_evaluate_math_func()]` when math sandbox is enabled, and later will also return skill tools and MCP tools. `ChatConsumer` then does `self.tools = self.doc_tools + get_external_tools(self) + [...]` instead of ad-hoc `if MATH_SANDBOX: self.tools.append(...)`.
- **Phase 3 tasks (later):** (1) MCP client module: connect to servers, list tools, invoke tool, map to `ToolResultDict`; (2) Skill loader: discover skills from config, call `get_tools`/`get_system_prompt_extra`, register with conversation; (3) Wire both into the same external-tool registration used for math.

---

## 3. Task Breakdown

### Task 1: Add math-sandbox Docker service (no Eigen yet)

**Files:**
- Create: `Dockerfile.math-sandbox`
- Modify: `docker-compose-development.yml`, `docker-compose-prod.yml` (or `docker-compose.yml`)

**Step 1:** Create `Dockerfile.math-sandbox` that:
- Uses a slim Python 3.12 image.
- Installs system dependencies: `libopenblas-dev` (or equivalent), build tools only if building numpy from source; otherwise use a scientific image or wheel that links OpenBLAS.
- Installs Python deps: `numpy`, `scipy` (and optionally `sympy` for symbolic math).
- Copies a small Flask/FastAPI or stdio runner that:
  - Accepts one request: code string.
  - Restricts imports via an allowlist (e.g. `math`, `numpy`, `scipy`, `sympy`).
  - Runs user code in a subprocess or restricted executor with `resource` module (RLIMIT_CPU, RLIMIT_AS) and a timeout (e.g. 5–10 s).
  - Returns JSON: `{"result": ...}` or `{"exception": "..."}`.
- Runs as non-root; no network needed inside container (only expose one port for the web app).

**Step 2:** Add service `math_sandbox` to compose files:
- Build from `Dockerfile.math-sandbox`.
- Expose port e.g. 5000 internally.
- Set resource limits: `mem_limit`, `cpus`, and optional `ulimits`.
- No `profiles` so it starts with default stack (or add `profiles: [math]` and document that users must use `--profile math` if you want it opt-in).

**Step 3:** Add `.env` entries:
- `MATH_SANDBOX_URL=http://math_sandbox:5000` (or similar).
- `MATH_SANDBOX_ENABLED=1`, `MATH_SANDBOX_TIMEOUT_SEC=10`.

**Verification:** `docker compose build math_sandbox && docker compose up -d math_sandbox` and curl `POST /eval` with `{"code": "import numpy as np; return float(np.linalg.det(np.eye(3)))"}` (adjust API to match). Expect `{"result": 1.0}`.

---

### Task 2: Implement sandbox runner (allowlist, timeout, RLIMIT)

**Files:**
- Create: `math_sandbox/app.py` (or `math_sandbox/main.py`)
- Create: `math_sandbox/sandbox.py` (execution with import allowlist and resource limits)

**Step 1:** Implement `sandbox.py`:
- Function `run_code(code: str, timeout_sec: float) -> dict`:
  - Option A: Run in subprocess with `resource.setrlimit(RLIMIT_CPU, RLIMIT_AS)` and `signal.alarm` or `threading.Timer` for timeout.
  - Option B: Use `concurrent.futures.ProcessPoolExecutor` with a single worker and `future.result(timeout=timeout_sec)`.
- Restrict imports: either (1) a custom import hook that only allows `math`, `numpy`, `scipy`, `sympy`, or (2) run in a minimal namespace where you inject only those modules and `__builtins__` restricted (e.g. no `open`, `exec`, `eval` with full builtins).
- Capture stdout/stderr; return last expression or a special variable (e.g. `__result__`) as the result. Serialize to JSON (convert numpy types to Python floats/ints/lists).
- Return `{"result": ...}` or `{"exception": "...", "traceback": "..."}`.

**Step 2:** Implement `app.py`:
- Minimal FastAPI or Flask app: one route `POST /eval` (or `/run`).
- Body: `{"code": "..."}`. Optional: `"timeout": 10`.
- Call `run_code(body["code"], timeout)` and return JSON response.
- Health route `GET /health` for compose healthcheck.

**Step 3:** Add `math_sandbox/requirements.txt`: `fastapi`, `uvicorn`, `numpy`, `scipy`, `sympy` (and optionally specify versions). No Django, no network libs beyond uvicorn.

**Verification:** Unit test `run_code` with safe code and with code that hits timeout or memory; test import allowlist blocks `os.system`.

---

### Task 3: Django client, LLM tool, and external-tool registration

**Files:**
- Create: `aquillm/aquillm/math_sandbox_client.py`
- Create: `aquillm/aquillm/tools/__init__.py` and `aquillm/aquillm/tools/external.py` (or equivalent single module for external tool collection)
- Modify: `aquillm/chat/consumers.py`
- Modify: `aquillm/settings.py` or `.env.example`

**Step 1:** Implement `math_sandbox_client.py`:
- Function `evaluate_math(code: str, timeout_sec: float | None = None) -> ToolResultDict`.
- If `MATH_SANDBOX_ENABLED` is false or `MATH_SANDBOX_URL` is empty, return `{"exception": "Math sandbox is not configured."}`.
- HTTP POST to `MATH_SANDBOX_URL/eval` with `{"code": code, "timeout": timeout_sec or default}`.
- Use `requests` or `httpx` with timeout; map response to `{"result": ...}` or `{"exception": ...}` in `ToolResultDict` format. Handle connection errors and timeouts.

**Step 2:** In `chat/consumers.py`, add a tool getter (e.g. `get_evaluate_math_func()`) that returns an `LLMTool`:
- Use `@llm_tool(for_whom='assistant', param_descs={"code": "Python code to run in a math sandbox. Use only math, numpy, scipy, sympy. Return the result as the last expression or assign to __result__."}, required=["code"])`.
- Tool implementation: call `evaluate_math(code)` and return the `ToolResultDict`.

**Step 3:** Introduce a single registration point for external tools (so MCP and skills can be added later without scattering conditionals):
- Create `aquillm/aquillm/tools/external.py` with a function `get_external_tools(consumer_or_context) -> list[LLMTool]`. For now it returns `[get_evaluate_math_func()]` when math sandbox is enabled, otherwise `[]`. Later this will also aggregate tools from MCP clients and skill loaders.
- In `ChatConsumer`, build tools as: `self.tools = self.doc_tools + get_external_tools(self) + [get_sky_subtraction_func(self), ...]` (i.e. replace any ad-hoc `if MATH_SANDBOX: append` with adding `get_external_tools(self)` once). Ensure `get_external_tools` receives whatever context it needs (e.g. the consumer instance for future skill/MCP context).

**Step 4:** Document in `.env.example`: `MATH_SANDBOX_URL`, `MATH_SANDBOX_ENABLED`, `MATH_SANDBOX_TIMEOUT_SEC`.

**Verification:** Start full stack with math_sandbox; open a chat and ask the assistant to compute something (e.g. "What is the determinant of a 3x3 identity matrix?"). Assistant should call the tool and return the result. No behavior change for non-math flows.

---

### Task 4: OpenBLAS in math-sandbox image

**Files:**
- Modify: `Dockerfile.math-sandbox`

**Step 1:** Ensure NumPy/SciPy use OpenBLAS:
- Option A: Use an official image that ships OpenBLAS (e.g. `python:3.12-slim` + `apt install libopenblas-dev` and then `pip install numpy scipy`; many wheels will pick up OpenBLAS if present).
- Option B: Use a scientific stack image (e.g. `continuumio/miniconda3` or `ghcr.io/.../science`) that is already linked to OpenBLAS.
- Document in Dockerfile comments that OpenBLAS is the BLAS backend for performance.

**Verification:** In sandbox, `import numpy as np; np.show_config()` (or similar) should show openblas; run a small benchmark (e.g. `np.dot(large, large)`) and confirm it runs.

---

### Task 5 (Phase 2): Eigen integration

**Files:**
- Create: `math_sandbox/eigen_bridge/` (C++ project or pybind11 module)
- Modify: `math_sandbox/sandbox.py` or sandbox Dockerfile

**Option A — pybind11 module:**
- Add Eigen (header-only) and pybind11 to the math-sandbox image.
- Build a small shared library that exposes Eigen operations (e.g. solve, inverse, SVD) to Python.
- In `sandbox.py` allowlist, add the custom module; LLM tool description can mention "Eigen-backed linear algebra when available."

**Option B — C++ stdio helper:**
- Build a small C++ binary that reads JSON from stdin (e.g. `{"op": "solve", "A": [[1,0],[0,1]], "b": [1,1]}`) and writes JSON to stdout.
- Sandbox service calls this binary via subprocess with timeout; only allow this specific binary, no arbitrary subprocess.
- Python sandbox then can offer a wrapper that calls the binary for selected operations.

**Step 1:** Implement one of the options; add tests for Eigen path (e.g. solve a 2x2 system).
**Step 2:** Document in tool description that advanced linear algebra may use Eigen for performance.

---

### Task 6: Hardening and observability

**Files:**
- Modify: `math_sandbox/sandbox.py`, `docker-compose-*.yml`, `.env.example`

**Step 1:** Add optional nsjail or seccomp profile for the sandbox process (if not using a separate container; if using container, Docker already provides isolation; optional second layer with read-only rootfs and no capabilities).
**Step 2:** Log sandbox requests (code hash or length, duration, success/failure) in Django for abuse detection; no logging of full code in production if privacy is a concern.
**Step 3:** Add `mem_limit` and `cpus` in compose; document recommended limits.

---

### Task 7 (Phase 3 — later): MCP client and server integration

**Purpose:** Allow AquiLLM to call tools exposed by MCP servers (stdio or HTTP). Math sandbox remains a separate service; MCP is for third-party or user-configured tool servers.

**Files:**
- Create: `aquillm/aquillm/mcp_client.py` (or `aquillm/mcp/` package)
- Modify: `aquillm/aquillm/tools/external.py`
- Modify: `.env.example`, `docker-compose-*.yml` (if MCP servers run as containers)

**Steps (summary):**
- Add MCP SDK dependency (e.g. `mcp` or official MCP Python client). Implement a thin wrapper that (1) connects to a configured list of MCP servers (stdio or HTTP), (2) lists tools from each server, (3) maps each MCP tool to an `LLMTool` whose `_function` calls the MCP client to invoke the tool and converts the response to `ToolResultDict`.
- In `get_external_tools()`, when `MCP_ENABLED` is set, instantiate the MCP client, fetch tools from all configured servers, and append the resulting `LLMTool` list to the return value (alongside math tools).
- Configuration: `MCP_ENABLED`, `MCP_SERVERS` (e.g. JSON or list of server configs: command/URL, env, timeouts). Document in README.

**Verification:** Configure one MCP server (e.g. a simple test server); open chat and trigger a tool that is provided by MCP; confirm result appears in conversation.

---

### Task 8 (Phase 3 — later): Skills loader and registration

**Purpose:** Load skills (bundles of tools and optional system-prompt extras) from a configured directory or config so that community or site-specific skills can extend the assistant without code changes.

**Files:**
- Create: `aquillm/aquillm/skills/loader.py` (or `aquillm/skills/`)
- Modify: `aquillm/aquillm/tools/external.py`
- Modify: `.env.example`, README

**Steps (summary):**
- Define a skill contract: a skill is a Python module or package that can export `get_tools(user, context) -> list[LLMTool]` and optionally `get_system_prompt_extra() -> str`. Context may include conversation, collections, etc.
- Implement a loader that reads `AQUILLM_SKILLS_DIR` (or a list of skill names from settings), imports the skill modules, and calls `get_tools` (and optionally augments system prompt with `get_system_prompt_extra`).
- In `get_external_tools()`, call the skills loader and append all skill-provided tools to the list returned (alongside math and MCP tools). Ensure stable ordering and no duplicate tool names (e.g. first registration wins or prefix by skill name).
- Configuration: `AQUILLM_SKILLS_ENABLED`, `AQUILLM_SKILLS_DIR` or `AQUILLM_SKILLS` (list). Document how to author a skill (one file or package, export `get_tools`).

**Verification:** Add a minimal skill (one tool) in the configured directory; restart; open chat and trigger that tool; confirm result.

---

## 4. Testing Strategy

- **Unit:** `math_sandbox/sandbox.py`: safe code returns result; timeout raises; disallowed import raises; memory-heavy code is killed.
- **Integration:** Django client: mock HTTP to sandbox and assert `ToolResultDict` shape; with real sandbox container, one E2E test that runs a simple numpy expression.
- **LLM tool:** Chat test that simulates assistant tool call `evaluate_math` with fixed input and checks stored tool message and result.

---

## 5. Documentation and .env

- **README:** Add a short "Math sandbox" section: what it does, that it runs in a container with NumPy/SciPy/OpenBLAS and optional Eigen, how to enable (`MATH_SANDBOX_ENABLED`, `MATH_SANDBOX_URL`), and that the LLM can use it via the evaluate_math tool. In the same or a follow-up section, note that the app is designed to support **skills** and **MCP servers** later (tools from external providers are registered in one place).
- **.env.example:** All new variables with comments (math sandbox now; MCP and skills when Phase 3 is implemented).

---

## 6. Dependencies

- **Existing:** AquiLLM already has `numpy` in `requirements.txt` (for the main app); the sandbox has its own `requirements.txt` and does not share the main app’s venv.
- **New (this plan):** In math-sandbox image: `numpy`, `scipy`, `sympy`, `fastapi`, `uvicorn`. Optional: Eigen (header-only), pybind11 (if Option A). Main app: `httpx` or `requests` for math-sandbox client (if not already present).
- **Future (Phase 3):** MCP Python SDK/client for Task 7; no new runtime deps for skills (Task 8) beyond existing Django/settings.

---

## 7. Security Summary

| Risk | Mitigation |
|------|-------------|
| Arbitrary code execution on host | Code runs only inside math-sandbox container; no host volume mount for code. |
| DoS (CPU/memory) | Docker limits + RLIMIT + request timeout; limit concurrent requests if needed. |
| Information disclosure | No network egress from sandbox; restrict imports; optional no-logging of code. |
| Malicious imports | Import allowlist (math, numpy, scipy, sympy, custom Eigen module only). |

---

## 8. Execution Order

**Phase 1–2 (this plan):**
1. Task 1 (Docker service + minimal stub that returns a fixed result).
2. Task 2 (sandbox runner with allowlist and timeout).
3. Task 3 (Django client + LLM tool **and** external-tool registration so math, MCP, and skills share one path).
4. Task 4 (OpenBLAS in image).
5. Task 5 (Eigen, phase 2).
6. Task 6 (hardening, logging, docs).

**Phase 3 (later — skills and MCP):**
7. Task 7 (MCP client and server integration).
8. Task 8 (Skills loader and registration).

---

Plan complete. Implement Phase 1–2 in the order above; each task can be committed separately. Use TDD where applicable (e.g. write client tests before implementing the client). The external-tool registration in Task 3 ensures that adding skills and MCP servers later only requires extending `get_external_tools()` and does not require scattering conditionals across the chat consumer.

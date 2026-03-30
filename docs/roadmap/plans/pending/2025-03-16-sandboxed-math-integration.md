# Sandboxed Math Integration (Python, Eigen, OpenBLAS) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable AquiLLM chat to perform advanced mathematics (numeric, linear algebra, symbolic) by executing user/LLM-requested code in a sandboxed environment using Python (NumPy/SciPy with OpenBLAS) and optional Eigen, without exposing the host or main app to arbitrary code execution. The design must also support adding **skills** and **MCP (Model Context Protocol) servers** later, so the math sandbox is the first instance of a consistent extensibility pattern. The sandbox must explicitly support two modes: (1) **computation mode** for calculator-style, exploratory, and research-oriented execution, and (2) **verification mode** for proof-like or high-confidence math outputs. In addition, math-derived answers should not go straight from generated code to user-visible finalization when the product is operating in verification mode: the plan now includes a distinct verification stage inspired by the generator/verifier/reviser workflow described in [arXiv:2602.10177](https://arxiv.org/pdf/2602.10177).

**Architecture:** A dedicated **math-sandbox** service runs in its own Docker container with strict resource and security limits. The Django app invokes it via HTTP (or a small RPC layer). The LLM gets a new low-level tool (e.g. `evaluate_math` or `run_python_math`) that sends code or expressions to the sandbox and returns structured results. This low-level tool can be used directly in **computation mode**. A separate verification layer can also treat those outputs as *candidate evidence*, rerun or cross-check them, and return a machine-readable verdict (`pass`, `warn`, or `fail`) before unattended finalization in **verification mode**. Eigen is integrated either as a Python-accessible C++ helper inside the same container (pybind11 or subprocess to a small binary) or as a separate microservice; OpenBLAS is used as the BLAS backend for NumPy/SciPy in the sandbox image. An **extensibility layer** is introduced so that tools can come from multiple sources (built-in, math sandbox, future MCP servers, future skills); the math sandbox is implemented as the first such external tool provider, with a clear contract and registration point that MCP and skills will use later.

**Tech Stack:** Docker, Python 3.12, NumPy, SciPy, OpenBLAS (via numpy/scipy build), optional Eigen (C++/pybind11 or standalone binary), timeout/RLIMIT-based sandboxing, optional nsjail/gVisor for stronger isolation. Future: MCP SDK/client, skills loader (config/DB or directory).

---

## 1. Scope and Constraints

- **Sandbox isolation:** No network egress, no filesystem write outside a temp dir, CPU/memory/time limits, no privileged syscalls.
- **Eigen:** Header-only C++; use for performance-critical linear algebra when NumPy is insufficient or when the LLM/user explicitly needs Eigen-style APIs. Options: (A) pybind11 wrapper in sandbox image, (B) small C++ HTTP/stdio helper in same container.
- **OpenBLAS:** Provide NumPy/SciPy built/linked against OpenBLAS in the sandbox image for fast BLAS/LAPACK.
- **Tool contract:** Input: code snippet or expression string; output: `ToolResultDict` (`result`, `exception`, optional `files`) so existing LLM tool handling in `chat/consumers.py` and `llm.py` works unchanged.
- **Dual-mode math flow:** The app must support both (1) direct sandbox execution for exploratory computation and (2) a separate verification pass for proof-like, publication-like, or otherwise high-confidence outputs. "The tool ran" is not sufficient evidence for a verified final answer, but verification should not be required for every raw calculator-style tool invocation.
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
| **Math verification coordinator** | Orchestrates a `generate -> verify -> revise` loop for math-heavy answers. Receives candidate code/results, runs verification checks, and decides whether to finalize, retry, or abstain. |
| **Verification report** | Machine-readable artifact describing claims checked, evidence used, rerun/cross-check outcomes, unresolved assumptions, and a final verdict (`pass`, `warn`, `fail`). |
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

- **Task 4 (Django client and LLM tool):** Implement the math tool so it is added via a single “external tools” collection point. For example: introduce `aquillm/aquillm/tools/external.py` (or a function in `consumers.py`) that returns `list[LLMTool]` for all enabled external providers; currently it returns `[get_evaluate_math_func()]` when math sandbox is enabled, and later will also return skill tools and MCP tools. `ChatConsumer` then does `self.tools = self.doc_tools + get_external_tools(self) + [...]` instead of ad-hoc `if MATH_SANDBOX: self.tools.append(...)`.
- **Phase 3 tasks (later):** (1) MCP client module: connect to servers, list tools, invoke tool, map to `ToolResultDict`; (2) Skill loader: discover skills from config, call `get_tools`/`get_system_prompt_extra`, register with conversation; (3) Wire both into the same external-tool registration used for math.

---

## 2c. Verification Loop for Math-Derived Answers

The most important idea to adopt from [arXiv:2602.10177](https://arxiv.org/pdf/2602.10177) is not the exact model stack, but the separation between **generation**, **verification**, and **revision**. In the paper, the agent uses distinct Generator, Verifier, and Reviser roles that iterate until the verifier approves or the attempt budget is exhausted. AquiLLM's math sandbox plan should adopt that same control structure for any answer that materially depends on sandboxed computation.

This verification loop is an additional operating mode layered on top of the sandbox, not a replacement for plain execution. The same sandbox should therefore support:

- **Computation mode:** run code, inspect outputs, and support exploratory research without invoking the verifier on every call.
- **Verification mode:** take a claim, derivation, or proof-like answer and require explicit checks plus a verdict before the system presents it as verified or high-confidence.

### Proposed AquiLLM flow

1. **Generate**
   - The assistant proposes code, assumptions, and an intended mathematical claim.
   - The sandbox executes the code and returns structured artifacts (`result`, stdout/stderr, serialized arrays, exception, timing, code hash).
2. **Verify**
   - A separate verifier path inspects the candidate claim and checks whether it is actually supported.
   - Required checks should include:
     - deterministic rerun of the same computation,
     - at least one independent cross-check when feasible (symbolic vs numeric, alternative formulation, invariant check, dimension/unit check, or small-case brute force),
     - assumption extraction and unresolved-dependency detection,
     - contradiction detection between prose claim and computed result.
3. **Revise**
   - If the verifier returns `fail` or a non-final `warn`, the system may revise the code/explanation and retry up to a configured limit.
4. **Finalize or abstain**
   - Only `pass` can finalize unattended.
   - `warn` may finalize only if the response clearly includes limitations, confidence, and missing checks, or if a later product policy explicitly allows it.
   - `fail` must not silently degrade into a confident answer; the assistant should say it could not verify the result.

### Verification artifact contract

Every math-heavy run should be able to emit a `MathVerificationReport` with fields such as:

- `claim_summary`
- `candidate_code_hash`
- `primary_result`
- `checks[]` with `name`, `status`, `details`, and optional supporting artifacts
- `assumptions[]`
- `open_questions[]`
- `attempt_index` / `max_attempts`
- `verdict` (`pass` | `warn` | `fail`)
- `recommended_user_facing_confidence`

This aligns well with the broader machine-readable report-card idea already proposed in `docs/specs/2026-03-26-langgraph-mcp-tools-orchestration-design.md`, while keeping the first implementation scoped to math.

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
- Include reproducibility-oriented metadata needed by the verifier: code hash, execution duration, timeout used, and a normalized serialization of the primary result.

**Step 2:** Implement `app.py`:
- Minimal FastAPI or Flask app: one route `POST /eval` (or `/run`).
- Body: `{"code": "..."}`. Optional: `"timeout": 10`.
- Call `run_code(body["code"], timeout)` and return JSON response.
- Health route `GET /health` for compose healthcheck.

**Step 3:** Add `math_sandbox/requirements.txt`: `fastapi`, `uvicorn`, `numpy`, `scipy`, `sympy` (and optionally specify versions). No Django, no network libs beyond uvicorn.

**Verification:** Unit test `run_code` with safe code and with code that hits timeout or memory; test import allowlist blocks `os.system`; confirm returned payload includes reproducibility metadata required for later verification.

---

### Task 3: Verification contract and verifier runner

**Files:**
- Create: `aquillm/aquillm/math_verification.py` (or `aquillm/lib/math/verification.py`)
- Modify: `aquillm/aquillm/math_sandbox_client.py` or adjacent shared schemas module
- Modify: `.env.example` or settings module for verification flags

**Step 1:** Define a verification schema:
- Introduce a typed candidate/report contract for math runs, e.g. `MathCandidate` and `MathVerificationReport`.
- Capture: claim summary, candidate code, primary result, extracted assumptions, sandbox metadata, check outcomes, and final verdict.

**Step 2:** Implement a verifier entrypoint:
- Function shape example: `verify_math_candidate(candidate: MathCandidate) -> MathVerificationReport`.
- Minimum checks:
  - rerun same code and compare normalized outputs,
  - run at least one independent cross-check when the candidate advertises enough structure to do so,
  - detect mismatches between computed value and prose explanation,
  - mark unverifiable cases explicitly instead of inferring success from the absence of an exception.

**Step 3:** Add retry policy and finalization rules:
- Introduce config such as `MATH_VERIFICATION_ENABLED=1`, `MATH_VERIFICATION_MAX_ATTEMPTS=2`, and `MATH_VERIFICATION_REQUIRE_PASS=1`.
- If a candidate fails verification, allow a bounded revise-and-retry loop.
- If the attempt budget is exhausted, return a structured abstention/failure message instead of a confident answer.

**Verification:** Add unit tests with canned candidates that produce each verdict:
- `pass`: deterministic rerun plus independent cross-check succeeds.
- `warn`: main result holds but assumptions remain unresolved.
- `fail`: rerun mismatch, contradiction, or unverifiable claim.

---

### Task 4: Django client, LLM tool, external-tool registration, and verified math flow

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
- Preserve verification metadata returned by the sandbox so the app can promote raw execution into a `MathCandidate` without recomputing context.

**Step 2:** In `chat/consumers.py`, add a tool getter (e.g. `get_evaluate_math_func()`) that returns an `LLMTool`:
- Use `@llm_tool(for_whom='assistant', param_descs={"code": "Python code to run in a math sandbox. Use only math, numpy, scipy, sympy. Return the result as the last expression or assign to __result__."}, required=["code"])`.
- Tool implementation: call `evaluate_math(code)` and return the `ToolResultDict`.
- Keep `evaluate_math` as the low-level execution primitive. Add a higher-level path in the chat runtime that can route candidate outputs through `verify_math_candidate(...)` when the request, policy, or UX calls for verification mode, while still allowing direct calculator-style execution in computation mode.

**Step 3:** Introduce a single registration point for external tools (so MCP and skills can be added later without scattering conditionals):
- Create `aquillm/aquillm/tools/external.py` with a function `get_external_tools(consumer_or_context) -> list[LLMTool]`. For now it returns `[get_evaluate_math_func()]` when math sandbox is enabled, otherwise `[]`. Later this will also aggregate tools from MCP clients and skill loaders.
- In `ChatConsumer`, build tools as: `self.tools = self.doc_tools + get_external_tools(self) + [get_sky_subtraction_func(self), ...]` (i.e. replace any ad-hoc `if MATH_SANDBOX: append` with adding `get_external_tools(self)` once). Ensure `get_external_tools` receives whatever context it needs (e.g. the consumer instance for future skill/MCP context).

**Step 4:** Document in `.env.example`: `MATH_SANDBOX_URL`, `MATH_SANDBOX_ENABLED`, `MATH_SANDBOX_TIMEOUT_SEC`.

**Step 5:** Add user-facing mode/finalization policy:
- In **computation mode**, the assistant may use sandbox outputs directly for exploratory work, while making clear when a result has not gone through verification.
- In **verification mode**, when verification returns `pass`, the assistant may present the result as verified.
- In **verification mode**, when verification returns `warn`, the assistant must surface limitations and the missing checks.
- In **verification mode**, when verification returns `fail`, the assistant should state that it could not verify the result and avoid overstating confidence.

**Verification:** Start full stack with math_sandbox; test both modes:
- In computation mode, ask the assistant to compute something exploratory (e.g. "numerically evaluate this expression") and confirm it can call the tool and return the result without invoking proof-style verification.
- In verification mode, ask the assistant to compute something with a checkable claim (e.g. "What is the determinant of a 3x3 identity matrix?") and confirm it calls the tool, produces a verification report, and only then returns the verified result. No behavior change for non-math flows.

---

### Task 5: OpenBLAS in math-sandbox image

**Files:**
- Modify: `Dockerfile.math-sandbox`

**Step 1:** Ensure NumPy/SciPy use OpenBLAS:
- Option A: Use an official image that ships OpenBLAS (e.g. `python:3.12-slim` + `apt install libopenblas-dev` and then `pip install numpy scipy`; many wheels will pick up OpenBLAS if present).
- Option B: Use a scientific stack image (e.g. `continuumio/miniconda3` or `ghcr.io/.../science`) that is already linked to OpenBLAS.
- Document in Dockerfile comments that OpenBLAS is the BLAS backend for performance.

**Verification:** In sandbox, `import numpy as np; np.show_config()` (or similar) should show openblas; run a small benchmark (e.g. `np.dot(large, large)`) and confirm it runs.

---

### Task 6 (Phase 2): Eigen integration

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

### Task 7: Hardening and observability

**Files:**
- Modify: `math_sandbox/sandbox.py`, `docker-compose-*.yml`, `.env.example`

**Step 1:** Add optional nsjail or seccomp profile for the sandbox process (if not using a separate container; if using container, Docker already provides isolation; optional second layer with read-only rootfs and no capabilities).
**Step 2:** Log sandbox requests (code hash or length, duration, success/failure) in Django for abuse detection; no logging of full code in production if privacy is a concern.
**Step 3:** Log verification outcomes (`pass`/`warn`/`fail`), attempt counts, and which checks were run; preserve enough metadata for audit/debugging without logging private code unnecessarily.
**Step 4:** Add `mem_limit` and `cpus` in compose; document recommended limits.

---

### Task 8 (Phase 3 — later): MCP client and server integration

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

### Task 9 (Phase 3 — later): Skills loader and registration

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
- **Unit:** `math_verification.py`: deterministic rerun, contradiction detection, independent cross-checks, verdict mapping, and bounded retry behavior.
- **Integration:** Django client: mock HTTP to sandbox and assert `ToolResultDict` shape; with real sandbox container, add one E2E test for computation mode and one for verification mode that emits a `pass` verification report.
- **LLM tool / orchestration:** Chat test that simulates assistant tool call `evaluate_math`, then verifies that raw compute remains available in computation mode, while finalization in verification mode is blocked on `fail`, caveated on `warn`, and normal on `pass`.

---

## 5. Documentation and .env

- **README:** Add a short "Math sandbox" section: what it does, that it runs in a container with NumPy/SciPy/OpenBLAS and optional Eigen, how to enable (`MATH_SANDBOX_ENABLED`, `MATH_SANDBOX_URL`), and that the LLM can use it via the evaluate_math tool. Make the two supported modes explicit: computation mode for exploratory/research work, and verification mode for checked math outputs. Add a follow-up section explaining the verification stage: verified answers are rerun/cross-checked before unattended finalization, and failures are surfaced as unverifiable results rather than hidden.
- **.env.example:** All new variables with comments (math sandbox now; verification flags now; MCP and skills when Phase 3 is implemented).

---

## 6. Dependencies

- **Existing:** AquiLLM already has `numpy` in `requirements.txt` (for the main app); the sandbox has its own `requirements.txt` and does not share the main app’s venv.
- **New (this plan):** In math-sandbox image: `numpy`, `scipy`, `sympy`, `fastapi`, `uvicorn`. Optional: Eigen (header-only), pybind11 (if Option A). Main app: `httpx` or `requests` for math-sandbox client (if not already present).
- **Possible helper deps:** If useful for typed report validation, add `pydantic` or equivalent schema tooling for `MathCandidate` / `MathVerificationReport`. Keep this optional unless the runtime already standardizes on one schema library.
- **Future (Phase 3):** MCP Python SDK/client for Task 8; no new runtime deps for skills (Task 9) beyond existing Django/settings.

---

## 7. Security Summary

| Risk | Mitigation |
|------|-------------|
| Arbitrary code execution on host | Code runs only inside math-sandbox container; no host volume mount for code. |
| DoS (CPU/memory) | Docker limits + RLIMIT + request timeout; limit concurrent requests if needed. |
| Information disclosure | No network egress from sandbox; restrict imports; optional no-logging of code. |
| Malicious imports | Import allowlist (math, numpy, scipy, sympy, custom Eigen module only). |
| Incorrect but plausible math answers | Separate verification stage with rerun/cross-checks; only `pass` finalizes unattended. |

---

## 8. Execution Order

**Phase 1–2 (this plan):**
1. Task 1 (Docker service + minimal stub that returns a fixed result).
2. Task 2 (sandbox runner with allowlist and timeout).
3. Task 3 (verification contract + verifier runner + verdict policy).
4. Task 4 (Django client + LLM tool + external-tool registration + verified math finalization path).
5. Task 5 (OpenBLAS in image).
6. Task 6 (Eigen, phase 2).
7. Task 7 (hardening, logging, docs).

**Phase 3 (later — skills and MCP):**
8. Task 8 (MCP client and server integration).
9. Task 9 (Skills loader and registration).

---

Plan complete. Implement Phase 1–2 in the order above; each task can be committed separately. Use TDD where applicable (e.g. write verifier tests before implementing the verifier, and client tests before implementing the client). The external-tool registration in Task 4 ensures that adding skills and MCP servers later only requires extending `get_external_tools()` and does not require scattering conditionals across the chat consumer. The new verification contract ensures that sandboxed math is not just executable, but auditable and bounded by an explicit pass/warn/fail gate before final answers are emitted.

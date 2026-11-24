# CadRelay

**Local-first CAD automation for repeatable mechanical assemblies.**

CadRelay turns YAML or JSON manifests into CadQuery solids, checks volume and interference with OpenCASCADE, and exports STEP and GLB. Define a part family once, then place its instances in a line, on a bolt circle, or with individual transforms.

- **Deterministic compiler** for shafts, flanges, sleeves, brackets, hollow frustums, and swept tubes.
- **MCP server** for batch compilation, assembly, and file inspection.
- **Local planning** with Ollama, plus optional direct cloud planning with a bounded repair pass.
- **Verification** for watertight solids, expected volumes and bounds, and BRep interference. Reports distinguish confirmed clearance from incomplete checks.

## Quick start

Use Python 3.12. Run these commands from the repository root in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m orchestrator.hybrid_orchestrator examples/drive_family/size_m.json --output workspace/drive_m
```

The example creates a 40-solid drive unit with STEP, GLB, and a verification report. Each invocation needs a new output directory.

For a local natural-language plan, start Ollama and use `--local-only` with `--prompt`. The default model is `qwen2.5:1.5b`. The manifest example can run without Ollama or a cloud key.

```powershell
ollama pull qwen2.5:1.5b
.\.venv\Scripts\python.exe -m orchestrator.hybrid_orchestrator --local-only --prompt "A 10 mm diameter pin, 16 mm tall, at the origin." --export-name pin --output workspace/pin
```

## Use as an MCP server

Configure your MCP client to launch the virtual environment's Python with `-m mcp_server.server` and set its working directory to the repository root. See the [Codex guide](CODEX.md) for manifest syntax and optional cloud configuration.

## Benchmark

The [drive family benchmark](benchmarks/drive_family_ab/report.md) qualified both CAD routes on three variants, with **40, 56, and 32 watertight solids**. In that experiment, the hybrid route used one Codex session instead of three and consumed **65.57% fewer uncached input plus output tokens**. The two derivative manifests were executed locally. Those figures describe this benchmark and include its recorded cache conditions.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The optional live-model test requires Ollama and `CAD_RUN_OLLAMA_TESTS=1`.

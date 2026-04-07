"""
Pipeline principal — Roda todos os steps em sequência.

Uso:
  python run_pipeline.py              # Roda tudo
  python run_pipeline.py --step 1     # Roda só o step 1
  python run_pipeline.py --step 2-3   # Roda steps 2 e 3
  python run_pipeline.py --max 5      # Limita a 5 episódios (para teste)
"""

import argparse
import os
import subprocess
import sys
import time

# Força UTF-8 no Windows para suportar emojis/acentos
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def run_step(step_num: int, script: str, extra_args: list[str] = None):
    """Executa um step do pipeline."""
    print()
    print("█" * 60)
    print(f"█  STEP {step_num} — {script}")
    print("█" * 60)
    print()

    cmd = [sys.executable, script] + (extra_args or [])
    start = time.time()

    result = subprocess.run(cmd, cwd=sys.path[0] or ".")
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ Step {step_num} falhou (código {result.returncode})")
        print(f"   Tempo: {elapsed:.0f}s")
        return False

    print(f"\n✅ Step {step_num} concluído em {elapsed:.0f}s")
    return True


def parse_step_range(step_str: str) -> tuple[int, int]:
    """Parse '2-3' → (2, 3) ou '2' → (2, 2)."""
    if "-" in step_str:
        parts = step_str.split("-")
        return int(parts[0]), int(parts[1])
    return int(step_str), int(step_str)


def main():
    parser = argparse.ArgumentParser(description="Pipeline G4 Podcast RAG")
    parser.add_argument("--step", type=str, default="1-4", help="Step(s) a executar: '1', '2-4', etc.")
    parser.add_argument("--max", type=int, default=0, help="Máximo de episódios (0=todos)")
    args = parser.parse_args()

    start_step, end_step = parse_step_range(args.step)
    extra = []
    if args.max > 0:
        extra = ["--max", str(args.max)]

    steps = {
        1: ("step1_fetch_episodes.py", []),
        2: ("step2_download_audio.py", extra),
        3: ("step3_transcribe.py", extra),
        4: ("step4_build_index.py", []),
    }

    total_start = time.time()
    print("=" * 60)
    print("🚀 G4 PODCAST RAG — PIPELINE")
    print(f"   Steps: {start_step} → {end_step}")
    if args.max:
        print(f"   Limite: {args.max} episódios")
    print("=" * 60)

    for step_num in range(start_step, end_step + 1):
        if step_num not in steps:
            print(f"⚠️  Step {step_num} não existe")
            continue

        script, step_extra = steps[step_num]
        success = run_step(step_num, script, step_extra)
        if not success:
            print(f"\n💀 Pipeline interrompido no step {step_num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print(f"🎉 PIPELINE CONCLUÍDO em {total_elapsed / 60:.1f} minutos")
    print()
    print("Próximo passo: Configurar o MCP Server no Claude Desktop.")
    print("Veja as instruções no README ou rode:")
    print("  python mcp_server.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

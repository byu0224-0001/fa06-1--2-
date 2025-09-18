import asyncio
import os
import sys
import nbformat
from nbclient import NotebookClient
try:
    # nbclient>=0.9 에서는 CellExecutionError가 exceptions로 분리됨
    from nbclient.exceptions import CellExecutionError  # type: ignore
except Exception:  # 하위 버전 호환
    try:
        from nbclient.client import CellExecutionError  # type: ignore
    except Exception:
        CellExecutionError = Exception  # 최후의 fallback


def execute_notebook(path: str) -> None:
    # Windows에서 ZMQ 경고 회피
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"[runner] Loading notebook: {path}")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
        record_timing=True,
    )
    try:
        print("[runner] Executing notebook from top to bottom...")
        client.execute()
        nbformat.write(nb, path)
        print("[runner] Execution completed successfully.")
    except CellExecutionError as e:
        # 실패한 셀의 에러를 그대로 출력하고 종료 코드 1로 종료
        print("[runner] Execution failed with CellExecutionError:\n", file=sys.stderr)
        print(e, file=sys.stderr)
        nbformat.write(nb, path)
        sys.exit(1)


if __name__ == "__main__":
    target = "최종_농산물_예측1.ipynb"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if not os.path.exists(target):
        print(f"Notebook not found: {target}", file=sys.stderr)
        sys.exit(2)
    execute_notebook(target)



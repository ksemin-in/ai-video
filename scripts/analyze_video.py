#!/usr/bin/env python3
"""
scripts/analyze_video.py
────────────────────────
Standalone CLI to run AI analysis on any video file.
Does NOT need the FastAPI server or database running.

Usage:
    python analyze_video.py --input traffic.mp4
    python analyze_video.py --input traffic.mp4 --speed-limit 80 --no-anpr
    python analyze_video.py --input traffic.mp4 --output my_output.mp4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make sure we can import from the backend
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Minimal env setup so config doesn't fail
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
os.environ.setdefault("REDIS_URL",    "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY",   "cli-secret")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import cv2

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="AI Traffic Video Analyzer — offline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",       required=True,  help="Input video path")
    parser.add_argument("--output",      default="",     help="Annotated video output path")
    parser.add_argument("--report",      default="",     help="JSON report output path")
    parser.add_argument("--speed-limit", type=float, default=60.0, help="Speed limit in km/h")
    parser.add_argument("--no-anpr",     action="store_true",      help="Skip ANPR (faster)")
    parser.add_argument("--skip-frames", type=int,   default=2,    help="Process every Nth frame")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        sys.exit(1)

    stem         = input_path.stem
    output_path  = args.output  or f"{stem}_annotated.mp4"
    report_path  = args.report  or f"{stem}_report.json"
    run_anpr     = not args.no_anpr

    # Quick video info
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    console.print(Panel(
        f"[bold cyan]AI Traffic Surveillance — Video Analyzer[/bold cyan]\n\n"
        f"Input   : [yellow]{input_path}[/yellow]\n"
        f"Output  : [yellow]{output_path}[/yellow]\n"
        f"Report  : [yellow]{report_path}[/yellow]\n"
        f"Video   : {total_frames} frames  ·  {fps:.1f} fps  ·  {w}×{h}\n"
        f"Duration: {total_frames/fps:.1f}s\n"
        f"ANPR    : {'enabled' if run_anpr else 'disabled'}\n"
        f"Speed limit: {args.speed_limit} km/h",
        border_style="cyan",
    ))

    # Override FRAME_SKIP via env
    os.environ["FRAME_SKIP"] = str(args.skip_frames)

    from app.api.routes.video_analytics import process_video_sync

    job_id    = f"cli-{int(time.time())}"
    start     = time.time()

    console.print("\n[cyan]Starting AI processing …[/cyan]")

    # Run (blocking — fine for CLI)
    try:
        report = process_video_sync(
            input_path=str(input_path),
            output_video_path=output_path,
            output_report_path=report_path,
            job_id=job_id,
            run_anpr=run_anpr,
            speed_limit=args.speed_limit,
        )
    except Exception as e:
        console.print(f"[red]Processing failed: {e}[/red]")
        sys.exit(1)

    elapsed = time.time() - start

    # ── Results ───────────────────────────────────────────────────
    console.print("\n")

    stats = Table(title="[bold green]Analysis Results[/bold green]", border_style="green")
    stats.add_column("Metric",      style="cyan",   no_wrap=True)
    stats.add_column("Value",       style="white")

    stats.add_row("Processing time",   f"{elapsed:.1f}s")
    stats.add_row("Unique vehicles",   str(report["unique_tracks"]))
    stats.add_row("ANPR reads",        str(len(report["anpr_reads"])))
    stats.add_row("Avg speed",         f"{report['avg_speed_kmh']} km/h")
    stats.add_row("Max speed",         f"{report['max_speed_kmh']} km/h")
    stats.add_row("Speeding events",   str(len(report["violations"]["speeding"])))
    stats.add_row("Watchlist hits",    str(len(report["violations"]["watchlist"])))

    console.print(stats)

    # Vehicle types
    if report["vehicle_types"]:
        vt = Table(title="Vehicle Types", border_style="blue")
        vt.add_column("Type");  vt.add_column("Count", style="yellow")
        for t, c in sorted(report["vehicle_types"].items(), key=lambda x: -x[1]):
            vt.add_row(t, str(c))
        console.print(vt)

    # ANPR sample
    if report["anpr_reads"]:
        console.print("\n[bold]ANPR Reads (first 10):[/bold]")
        at = Table(border_style="yellow")
        at.add_column("Plate"); at.add_column("Conf"); at.add_column("Time"); at.add_column("Watchlisted")
        for r in report["anpr_reads"][:10]:
            wl = "[red]YES[/red]" if r["watchlisted"] else "no"
            at.add_row(r["plate"], f"{r['confidence']:.2f}", f"{r['time_s']:.1f}s", wl)
        console.print(at)

    # Violations
    if report["violations"]["speeding"]:
        console.print(f"\n[red]⚠  {len(report['violations']['speeding'])} speeding violations detected[/red]")
        for v in report["violations"]["speeding"][:5]:
            console.print(f"   Track #{v['track_id']}  {v['speed_kmh']} km/h  at {v['time_s']}s")

    console.print(f"\n[green]✓ Annotated video:[/green] {output_path}")
    console.print(f"[green]✓ Full JSON report:[/green] {report_path}")
    console.print(f"[green]✓ Done in {elapsed:.1f}s[/green]\n")


if __name__ == "__main__":
    main()

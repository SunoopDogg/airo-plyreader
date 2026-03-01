"""
Standalone viewer for previous pipeline results.

Scans the output directory for previous runs and opens the selected
run's PLY files in Open3D viewer windows.
"""

from pathlib import Path


def list_run_dirs() -> list[Path]:
    """Scan output directory for previous run subdirectories."""
    from .config import OUTPUT_DIR
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_dir():
        return []
    return sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )


def title_from_filename(filename: str) -> str:
    """Derive a display title from a PLY filename.

    Example: 'downsampled_points.ply' -> 'Downsampled Points'
    """
    return filename.removesuffix(".ply").replace("_", " ").title()


def prompt_run_selection(run_dirs: list[Path]) -> Path | None:
    """Let the user pick a run directory interactively."""
    print("\nAvailable runs:")
    for i, d in enumerate(run_dirs, 1):
        ply_count = len(list(d.glob("*.ply")))
        print(f"  {i}) {d.name}  ({ply_count} PLY files)")
    print()

    while True:
        try:
            choice = int(input(f"Select run number (1-{len(run_dirs)}): "))
            if 1 <= choice <= len(run_dirs):
                selected = run_dirs[choice - 1]
                print(f"Selected: {selected.name}\n")
                return selected
        except ValueError:
            pass
        except EOFError:
            print("\nNo input available.")
            return None
        print(f"Please enter a number between 1 and {len(run_dirs)}.")


def main() -> None:
    """Main entry point for the standalone viewer."""
    run_dirs = list_run_dirs()

    if not run_dirs:
        print("No previous runs found in output/ directory.")
        return

    selected_dir = prompt_run_selection(run_dirs)
    if selected_dir is None:
        return

    # Build targets from all PLY files in the selected directory
    ply_files = sorted(selected_dir.glob("*.ply"))

    if not ply_files:
        print(f"No PLY files found in {selected_dir.name}/")
        return

    targets = [
        (title_from_filename(f.name), str(f))
        for f in ply_files
    ]

    print(f"Opening {len(targets)} file(s):")
    for title, path in targets:
        print(f"  - {title}")
    print()

    from .visualization.visualization import launch_all_viewers
    launch_all_viewers(targets)


if __name__ == "__main__":
    main()

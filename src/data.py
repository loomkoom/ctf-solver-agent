import os
import shutil
from pathlib import Path

def cleanup_tree(base_path):
    p = Path(base_path)
    for file in p.rglob("*"):
        # 1. Rename paths with spaces or special chars
        if ' ' in file.name or '=' in file.name:
            new_name = file.name.replace(' ', '-').replace('=', '')
            new_path = file.parent / new_name
            os.rename(file, new_path)
            file = new_path

        # 2. Move non-text artifacts to a parallel folder
        if file.is_file() and file.suffix not in ['.md', '.txt', '.py']:
            # Define target in 'data/artifacts'
            relative_path = file.relative_to(p)
            target = (Path(base_path) / Path("../../artifacts") / relative_path).resolve()
            print(f"Moving artifact: {file} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(target))

if __name__ == "__main__":
    data_path = (Path.cwd() / "../data/knowledge_base/writeups").resolve()
    print(f"Cleaning up data tree at: {data_path}")
    cleanup_tree(data_path)

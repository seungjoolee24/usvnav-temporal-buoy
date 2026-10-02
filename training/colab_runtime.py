"""Source loading and recoverable Drive backups for the temporal Colab notebook."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import threading
import time
import zipfile


def extract_source_bundle(payload, destination):
    """Verify allowlisted source bytes before extracting a manual upload."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read('_source_manifest.json'))
        if manifest.get('format') != 'usvnav-colab-source/1':
            raise ValueError('Use the ZIP produced by training.package_colab')
        expected = set(manifest['files']) | {'_source_manifest.json'}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError('Unexpected or duplicate ZIP entries')
        if sum(item.file_size for item in archive.infolist()) > 100_000_000:
            raise ValueError('Source archive is too large')
        for item in archive.infolist():
            name = item.filename
            if ('\\' in name or ':' in name or PurePosixPath(name).is_absolute()
                    or '..' in PurePosixPath(name).parts or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Invalid source path')
            if name != '_source_manifest.json':
                content, record = archive.read(name), manifest['files'][name]
                if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
                    raise ValueError(f'Source hash mismatch: {name}')
        destination.mkdir(parents=True)
        archive.extractall(destination)
    return manifest


def approve_curriculum_copy(source, destination, *, confirmed):
    """Record explicit notebook approval in a runtime copy, preserving source."""
    if confirmed is not True:
        raise ValueError('Review the environment images and set CONFIRM_ENVIRONMENT=True')
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    from training.collect_perception import course_identity
    from usvnav.coursefile import from_dict, validate
    plan = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    copies = {}
    for split in ('train', 'val', 'test', 'review'):
        for row in plan[split]:
            name = row['file']
            if Path(name).name != name:
                raise ValueError('Course name must be a plain filename')
            content = (source / name).read_bytes()
            course = from_dict(json.loads(content))
            if course_identity(course) != row['sha256'] or validate(course):
                raise ValueError(f'Changed or invalid course: {name}')
            copies[name] = content
    destination.mkdir(parents=True)
    for name, content in copies.items():
        (destination / name).write_bytes(content)
    plan['metadata'].update(approval_status='approved_by_user',
        approved_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest())
    (destination / 'manifest.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return destination / 'manifest.json'


def sync_run(source, destination):
    """Copy small run artifacts; skip partial checkpoints and bulky live datasets."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        return 0
    count = 0
    for path in source.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix not in {
                '.zip', '.json', '.jsonl', '.csv', '.md', '.png', '.py', '.txt'}:
            continue
        if '.pending.' in path.name or '.copying' in path.name:
            continue
        try:
            content = path.read_bytes()
            if path.suffix == '.zip' and not zipfile.is_zipfile(io.BytesIO(content)):
                continue
            if path.suffix == '.json' and path.parent.name != 'logs':
                json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.read_bytes() == content:
            continue
        temporary = target.with_name(target.name + '.copying')
        temporary.write_bytes(content)
        os.replace(temporary, target)
        count += 1
    return count


def run_with_backup(command, *, project_root, run_dir, backup_dir, interval_s=30):
    """Stream a CLI run and back up while it trains; preserve results on errors."""
    project_root, run_dir, backup_dir = map(Path, (project_root, run_dir, backup_dir))
    if run_dir.exists() or backup_dir.exists():
        raise FileExistsError('Choose fresh local and Drive run directories')
    if interval_s <= 0:
        raise ValueError('Backup interval must be positive')
    process = subprocess.Popen(command, cwd=project_root, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    stopped, failures = threading.Event(), []

    def backup_loop():
        while not stopped.wait(interval_s):
            try:
                sync_run(run_dir, backup_dir)
            except OSError as exc:
                failures.append(str(exc))
                print('Drive backup failed; local run continues:', exc, flush=True)

    worker = threading.Thread(target=backup_loop, daemon=True)
    worker.start()
    started = time.perf_counter()
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        stopped.set()
        worker.join()
        process.stdout.close()
        sync_run(run_dir, backup_dir)
    if code:
        raise subprocess.CalledProcessError(code, command)
    return dict(wall_s=time.perf_counter()-started, backup_dir=str(backup_dir),
                transient_backup_failures=len(failures))
